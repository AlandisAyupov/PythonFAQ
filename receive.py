import os
import ssl
from imap_tools import MailBox
from dotenv import load_dotenv

load_dotenv() 

# Set up the IMAP connection
context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.set_ciphers('DEFAULT@SECLEVEL=1')

# Iterate through each email message and print its contents
with MailBox('rci.rutgers.edu', ssl_context=context).login(os.getenv("EMAIL"), os.getenv("PASS"), 'INBOX') as mailbox:
    for msg in mailbox.fetch():
        print(msg.from_)
        print(msg.text)
    
