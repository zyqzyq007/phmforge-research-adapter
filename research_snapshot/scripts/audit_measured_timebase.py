"""Bounded raw-signal audit; no class learning, target scoring, or data mutation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import time
import zlib

import numpy as np

from audit_program_evidence import read, save, sha, utc


def events_vector(x, low, high):
    indices = np.flatnonzero((x <= low) | (x >= high))
    state = x[indices] >= high
    return indices[1:][(~state[:-1]) & state[1:]]


def events_loop(x, low, high):
    armed = False
    out = []
    for i, value in enumerate(x):
        if value <= low:
            armed = True
        elif armed and value >= high:
            out.append(i)
            armed = False
    return np.asarray(out, dtype=int)


def one(item, cfg):
    root = Path(cfg['source_archive'])
    path = root / 'records' / item['unit_id'] / 'record.csv'
    receipt = read(path.with_name('RECEIPT.json'))
    raw = path.read_bytes()
    import hashlib
    assert hashlib.sha256(raw).hexdigest() == receipt['sha256']
    assert len(raw) == item['bytes']
    assert f'{zlib.crc32(raw):08x}' == item['CRC32']
    delimiter = ';' if b';' in raw.split(b'\n', 1)[0] else ','
    array = np.loadtxt(io.BytesIO(raw), delimiter=delimiter, usecols=(0, 1))
    times, signal = array.T
    assert np.isfinite(array).all() and np.all(np.diff(times) > 0)
    fs = (len(times) - 1) / (times[-1] - times[0])
    qlow, qhigh = np.quantile(signal, cfg['quantiles'])
    assert qhigh > qlow
    low = qlow + cfg['schmitt_levels'][0] * (qhigh - qlow)
    high = qlow + cfg['schmitt_levels'][1] * (qhigh - qlow)
    indices = events_vector(signal, low, high)
    independent = events_loop(signal, low, high)
    assert np.array_equal(indices, independent)
    result = dict(unit=item['unit_id'], role=item['role'], nominal_hz=item['recorded_rotation_Hz'],
                  rows=len(times), raw_sha256=receipt['sha256'], raw_crc32=item['CRC32'],
                  fs_hz=float(fs), duration_s=float(times[-1] - times[0]),
                  uniform_grid_max_s=float(np.max(np.abs(times - times[0] - np.arange(len(times)) / fs))),
                  signal_quantiles=np.quantile(signal, [0, .01, .25, .5, .75, .99, 1]).tolist(),
                  schmitt_levels=[float(low), float(high)], event_indices=indices.tolist(),
                  pulse_count=len(indices), independent_events_exact=True)
    if len(indices) >= 3:
        dt = np.diff(times[indices])
        pulse_hz = (len(indices) - 1) / (times[indices[-1]] - times[indices[0]])
        index_hz = (len(indices) - 1) * fs / (indices[-1] - indices[0])
        assert abs(pulse_hz - index_hz) < 1e-5
        result.update(pulse_hz=float(pulse_hz), pulse_to_filename_ratio=float(pulse_hz / item['recorded_rotation_Hz']),
                      index_formula_hz=float(index_hz), median_period_s=float(np.median(dt)),
                      period_cv=float(np.std(dt) / np.mean(dt)),
                      period_min_s=float(dt.min()), period_max_s=float(dt.max()))
    transformed = np.abs(np.fft.rfft((signal - signal.mean()) * np.hanning(len(signal)))) ** 2
    frequencies = np.fft.rfftfreq(len(signal), d=1 / fs)
    band = np.flatnonzero((frequencies >= cfg['fft_band_hz'][0]) & (frequencies <= cfg['fft_band_hz'][1]))
    peak = band[np.argmax(transformed[band])]
    result['spectral_peak_hz'] = float(frequencies[peak])
    result['spectrum_resolution_hz'] = float(fs / len(signal))
    result['spectral_peak_to_filename_ratio'] = float(frequencies[peak] / item['recorded_rotation_Hz'])
    return result


def main(root):
    cfg = read(root / 'PROTOCOL401.json')
    freeze = read(root / 'CODE_FREEZE401.json')
    assert sha(Path(__file__)) == freeze['script_sha256']
    assert sha(Path(__file__).with_name('audit_program_evidence.py')) == freeze['shared_helpers_sha256']
    assert sha(root / 'PROTOCOL401.json') == freeze['protocol_sha256']
    assert not (root / 'STARTED401.json').exists()
    source = Path(cfg['source_archive'])
    receipt = read(source / 'ARCHIVE_VERIFIED.json')
    manifest = source / receipt['manifest_filename']
    assert sha(manifest) == cfg['source_manifest_sha256'] == receipt['manifest_sha256']
    index = {x['path']: x for x in read(manifest)['files']}
    for name in ['PROTOCOL346.json', 'SOURCE_RECOVERY_RESULT.json']:
        assert sha(source / name) == index[name]['sha256']
    rows = read(source / 'PROTOCOL346.json')['selected_records']
    assert len(rows) == cfg['expected_records']
    for row in rows:
        name = 'records/' + row['unit_id'] + '/RECEIPT.json'
        assert sha(source / name) == index[name]['sha256']
    save(root / 'STARTED401.json', dict(utc=utc(), status='running'))
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=cfg['workers']) as executor:
        records = list(executor.map(lambda row: one(row, cfg), rows))
    usable = [r for r in records if 'pulse_hz' in r]
    ratios = [r['pulse_to_filename_ratio'] for r in usable]
    result = dict(utc=utc(), status='completed', seconds=time.monotonic() - start,
                  records=records, count=len(records), periodic_records=len(usable),
                  pulse_ratio_median=float(np.median(ratios)) if ratios else None,
                  pulse_ratio_range=[min(ratios), max(ratios)] if ratios else None,
                  pulse_close_to_one_count=sum(abs(x - 1) <= cfg['relative_tolerance'] for x in ratios),
                  pulse_close_to_clock_ratio_count=sum(abs(x - 51200 / 50000)
                                                      <= cfg['relative_tolerance'] for x in ratios),
                  spectral_close_to_one_count=sum(abs(r['spectral_peak_to_filename_ratio'] - 1)
                                                 <= cfg['relative_tolerance'] for r in records),
                  ppr_verified=False, agent_causal_claim_established=False,
                  source_files_modified=False, new_LLM_calls=0, new_PHM_fits=0,
                  new_PHM_forward_calls=0, reserved_confirmation_opened=False)
    save(root / 'RESULT401.json', result)
    save(root / 'INDEPENDENT_AUDIT401.json', dict(utc=utc(), passed=True,
         exact_event_checks=len(records), independent_time_formula_checks=len(usable),
         checks=3 * len(records) + len(usable), raw_sha_crc_checks=2 * len(records),
         note='Loop versus filtered state changes; elapsed timestamp versus index-rate formula. FFT is a corroboration, not a PPR calibration.'))
    save(root / 'TERMINAL401.json', dict(utc=utc(), status='completed', new_LLM_calls=0,
                                       new_PHM_fits=0, new_PHM_forward_calls=0))
    print({k: v for k, v in result.items() if k != 'records'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve())
