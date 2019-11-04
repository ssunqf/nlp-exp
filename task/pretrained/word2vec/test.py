#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-


import libtorrent as lt
import time
import tempfile

ses = lt.session()
params = {
    'save_path': tempfile.mkdtemp(),
    'storage_mode': lt.storage_mode_t(2),
    'auto_managed': True,
    'file_priorities': [0]*5
}

handle = lt.add_magnet_uri(ses, "magnet:?xt=urn:btih:9fea16aff4ece16e04f98321668a265f0fd22b7e&dn=archlinux-2017.08.01-x86_64.iso&tr=udp://tracker.archlinux.org:6969&tr=http://tracker.archlinux.org:6969/announce", params)
while(not handle.has_metadata()):
    time.sleep(1)

print(handle.get_torrent_info().name())