import os
import time

import nest_asyncio
from pupil_labs.realtime_api.simple import discover_one_device
from datetime import datetime
command = 'cmd /k "C:/Users/TUTTI/Desktop/PROVA_EYE_TRACKER/script.bat"'

#time.sleep(2)
# multiprocess
nest_asyncio.apply()
device = discover_one_device()

print(f"Phone IP address: {device.phone_ip}")
print(f"Phone name: {device.phone_name}")
print(f"Battery level: {device.battery_level_percent}%")
print(f"Free storage: {device.memory_num_free_bytes / 1024**3:.1f} GB")
print(f"Serial number of connected glasses: {device.module_serial}")
recording_id = device.recording_start()
print(f"Started recording with id {recording_id}")
now = datetime.now()
name = str(now)
#device.send_event(name)
#event_timestamp_unix_ns=time.time_ns()
#device.send_event("unix time", event_timestamp_unix_ns=event_timestamp_unix_ns)
device.send_event(str(time.time_ns()))
now = datetime.now()
name = str(now)
time = name.split('.')
name = name.split('.')[-1]
with open('./'+name+'.txt','w') as f:
    f.write(str(time))
f.close()
os.system(command)
os.system('cmd /k "exit"')
#time.sleep(5)
device.recording_stop_and_save()
