import uuid
import base64
import boto3
import requests

import requests
url = "https://www.fast2sms.com/dev/bulk"
payload = "sender_id=FSTSMS&message="+"Baby started Crying"+"&language=english&route=p&numbers=9963611235"
headers = {
 'authorization': "vijZsnUAxX1NmTElCFfu0QWwa8zd3pbSR6I2VOHPrtyqG9koL7LZoiYp01D2EWc8vklwOPIzUqfmgN6A",
 'Content-Type': "application/x-www-form-urlencoded",
 'Cache-Control': "no-cache",
 }
response = requests.request("POST", url, data=payload, headers=headers)
print(response.text)
