import os
import ssl
from imap_tools import MailBox
from dotenv import load_dotenv

load_dotenv() 

# MAIL - Credit to stackoverflow user https://stackoverflow.com/questions/5632713/getting-n-most-recent-emails-using-imap-and-python

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.set_ciphers('DEFAULT@SECLEVEL=1')
with MailBox('rci.rutgers.edu', ssl_context=context).login(os.getenv("EMAIL"), os.getenv("PASS"), 'INBOX') as mailbox:
    for msg in mailbox.fetch(reverse=True):
        mailbox.delete([msg.uid for msg in mailbox.fetch()])