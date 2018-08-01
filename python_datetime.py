# get China Time
from datetime import datetime
arrow.get(datetime.now(), 'Asia/Shanghai')


# change from utc time to Chinese time
from dateutil import tz
from datetime import datetime

from_zone = tz.gettz('UTC') # UTC Zone --> UTC
to_zone = tz.gettz('Asia/Shanghai')# China Zone --> PRC

utc = datetime.utcnow()
utc = utc.replace(tzinfo=from_zone)
local = utc.astimezone(to_zone)
print(datetime.strftime(local, "%Y-%m-%d %H:%M:%S"))


