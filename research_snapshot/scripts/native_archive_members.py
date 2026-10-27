"""Read measured-data archive members through the installed system libarchive.

No extraction, executable invocation, or temporary directory is required.
"""
import ctypes as C
import ctypes.util


def members(path, maximum_member_bytes=64*1024**2):
    library=C.CDLL(ctypes.util.find_library('archive'))
    def bind(name,result,args):
        method=getattr(library,name);method.restype=result;method.argtypes=args;return method
    new=bind('archive_read_new',C.c_void_p,[])
    filters=bind('archive_read_support_filter_all',C.c_int,[C.c_void_p])
    formats=bind('archive_read_support_format_all',C.c_int,[C.c_void_p])
    opened=bind('archive_read_open_filename',C.c_int,[C.c_void_p,C.c_char_p,C.c_size_t])
    next_header=bind('archive_read_next_header',C.c_int,[C.c_void_p,C.POINTER(C.c_void_p)])
    pathname=bind('archive_entry_pathname',C.c_char_p,[C.c_void_p])
    size=bind('archive_entry_size',C.c_int64,[C.c_void_p])
    read=bind('archive_read_data',C.c_int64,[C.c_void_p,C.c_void_p,C.c_size_t])
    error=bind('archive_error_string',C.c_char_p,[C.c_void_p])
    free=bind('archive_read_free',C.c_int,[C.c_void_p])
    handle=new();filters(handle);formats(handle)
    try:
        if isinstance(path, (list, tuple)):
            if not path:
                raise ValueError('At least one native archive volume is required')
            opened_many=bind('archive_read_open_filenames',C.c_int,
                             [C.c_void_p,C.POINTER(C.c_char_p),C.c_size_t])
            names=(C.c_char_p*(len(path)+1))(*[str(p).encode() for p in path],None)
            status=opened_many(handle,names,1024**2)
        else:
            status=opened(handle,str(path).encode(),1024**2)
        if status!=0:raise RuntimeError(error(handle))
        entry=C.c_void_p();buffer=C.create_string_buffer(1024**2)
        while True:
            code=next_header(handle,C.byref(entry))
            if code==1:break
            if code!=0:raise RuntimeError(error(handle))
            name=pathname(entry).decode();length=size(entry)
            if length<0 or length>maximum_member_bytes:raise ValueError('Archive member size outside bound: '+name)
            blocks=[];received=0
            while True:
                n=read(handle,buffer,len(buffer))
                if n<0:raise RuntimeError(error(handle))
                if n==0:break
                received+=n
                if received>maximum_member_bytes:raise ValueError('Archive member exceeds declared bound')
                blocks.append(buffer.raw[:n])
            if received!=length:raise ValueError('Archive member byte count disagrees: '+name)
            yield name,b''.join(blocks)
    finally:
        free(handle)
