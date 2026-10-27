"""Read selected public H5 records through the ordinary anonymous web UI.

No cookies, authentication stores, request headers, or response bodies are read.
The transport observer records only paths, status codes, and byte counts.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    root = args.run.resolve()
    config = json.loads(args.config.read_text())
    assert args.attempt >= 1
    out = root / ('public_collection' if args.attempt == 1 else f'public_collection_{args.attempt:02}')
    out.mkdir(exist_ok=False)
    downloads = root / 'public_downloads'
    downloads.mkdir(exist_ok=True)
    driver_path = Path(config['driver_path'])
    assert hashlib.sha256(driver_path.read_bytes()).hexdigest() == config['driver_sha256']
    opt = Options()
    opt.headless = True
    for key, value in {'browser.cache.disk.enable': False, 'browser.download.folderList': 2,
                       'browser.download.dir': str(downloads),
                       'browser.helperApps.neverAsk.saveToDisk': 'application/pdf,application/octet-stream,image/tiff',
                       'pdfjs.disabled': True}.items():
        opt.set_preference(key, value)
    driver = webdriver.Firefox(service=Service(executable_path=str(driver_path), log_path='/dev/null'), options=opt)
    driver.set_page_load_timeout(60)
    seq = 0
    records = []

    def persist():
        (out / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')

    def snapshot(label):
        nonlocal seq
        seq += 1
        rec = driver.execute_script("""return {
          body:document.body.innerText,
          buttons:Array.from(document.querySelectorAll('button,a[role=button]')).filter(e=>e.offsetParent!==null).map(e=>({text:e.innerText,cls:e.className})),
          transport:window.__phmSourceTransport||[]};""")
        rec.update(label=label, utc=utc())
        (out / f'{seq:03}.json').write_text(json.dumps(rec, ensure_ascii=False, indent=2) + '\n')
        driver.save_screenshot(str(out / f'{seq:03}.png'))
        print(json.dumps({'snapshot': seq, 'label': label, 'visible_tail': rec['body'][-450:],
                          'transport_tail': rec['transport'][-3:]}, ensure_ascii=False), flush=True)
        return rec

    def observe_transport():
        driver.execute_script("""if(!window.__phmSourceTransport){
          window.__phmSourceTransport=[];
          const oldOpen=XMLHttpRequest.prototype.open;
          XMLHttpRequest.prototype.open=function(method,url){
            const u=new URL(url,location.href);
            const rec={method:method,host:u.host,path:u.pathname,started:Date.now()};
            this.addEventListener('progress',e=>{rec.loaded=e.loaded;rec.total=e.lengthComputable?e.total:null;});
            this.addEventListener('loadend',()=>{rec.status=this.status;rec.elapsed_ms=Date.now()-rec.started;window.__phmSourceTransport.push(rec);});
            return oldOpen.apply(this,arguments);
          };
        }""")

    def click_text(text, occurrence=0):
        candidates = driver.find_elements(By.XPATH, "//*[self::button or self::a or @role='menuitem' or @role='option'][normalize-space()=" + json.dumps(text) + "]")
        visible = [e for e in candidates if driver.execute_script('return arguments[0].offsetParent!==null', e)]
        if occurrence >= len(visible):
            raise ValueError('Requested public UI action not visible: ' + text)
        visible[occurrence].click()

    def wait_css(selector, seconds):
        started = time.monotonic()
        while time.monotonic() - started < seconds:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if elements:
                return elements
            time.sleep(1)
        raise TimeoutError('UI readiness: ' + selector)

    try:
        for target in config['tasks']:
            rec = dict(target, status='running', started_utc=utc(), downloads=[])
            records.append(rec)
            persist()
            try:
                try:
                    driver.get('https://rrcsearch3.neubus.com/search-profile?profileId=7')
                except TimeoutException:
                    # Navigation observation timed out; keep inspecting this same page.
                    rec['navigation_observation_timed_out'] = True
                field = wait_css('#search_field_uic_no', config['navigation_readiness_seconds'])[0]
                observe_transport()
                field.clear()
                field.send_keys(target['uic'])
                click_text('Search')
                wait_css('button.table-cell-button', 60)
                snapshot('query_' + target['uic'])
                click_text('clickable ' + target['date'], target['occurrence'])
                wait_css('.nde-view-record-item-name', 60)
                snapshot('record_' + target['uic'] + '_' + target['date'].replace('/', '-'))
                names = [e.text.strip() for e in driver.find_elements(By.CSS_SELECTOR, '.nde-view-record-item-name')]
                rec['visible_document_names'] = names
                for name in target['documents']:
                    if name not in names:
                        raise ValueError('Requested file not listed in public record')
                    destination = downloads / name
                    if destination.exists():
                        raise FileExistsError('Refuse ambiguous existing download: ' + name)
                    icons = driver.find_elements(By.CSS_SELECTOR, 'button.v-btn--icon.v-btn--round')
                    icons[names.index(name)].click()
                    time.sleep(1)
                    snapshot('download_menu_' + name)
                    click_text('Download')
                    started = time.monotonic()
                    heartbeat = started
                    complete = False
                    while time.monotonic() - started < config['download_wait_seconds']:
                        if destination.exists() and destination.stat().st_size > config['single_source_max_MiB'] * 1024**2:
                            raise ValueError('Source size exceeds frozen limit')
                        if destination.exists() and destination.stat().st_size > 0 and not any(downloads.glob('*.part')):
                            before = destination.stat().st_size
                            time.sleep(1)
                            if destination.stat().st_size == before:
                                complete = True
                                break
                        if time.monotonic() - heartbeat >= 40:
                            snapshot('download_pending_' + name)
                            heartbeat = time.monotonic()
                        time.sleep(1)
                    entry = {'file_label': name, 'elapsed_seconds': time.monotonic() - started,
                             'status': 'downloaded_pending_preflight' if complete else 'delivery_not_completed_within_budget'}
                    if complete:
                        entry.update(file=str(destination.relative_to(root)), bytes=destination.stat().st_size,
                                     sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
                    rec['downloads'].append(entry)
                    persist()
                    snapshot('download_result_' + name)
                    if not complete:
                        raise TimeoutError('Document delivery incomplete; stop this record before another download')
                rec['status'] = 'completed_UI_attempt'
            except Exception as exc:
                rec.update(status='unavailable', error_type=type(exc).__name__, error=str(exc)[:400])
                try:
                    snapshot('unavailable_' + target['uic'])
                except Exception as capture:
                    rec['snapshot_error_type'] = type(capture).__name__
            rec['ended_utc'] = utc()
            persist()
            print(json.dumps(rec, ensure_ascii=False), flush=True)
    finally:
        driver.quit()
        (out / 'closed.json').write_text(json.dumps({'closed_utc': utc()}) + '\n')


if __name__ == '__main__':
    main()
