import datetime
import pytz
print(' '.join(pytz.country_timezones('es')))
today = datetime.datetime.now(pytz.timezone("Europe/Madrid"))
print(today.date())
print(today.strftime("%H:%M:%S"))