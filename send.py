import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv() 

# Connect to the SMTP server
smtp_server = 'mx.farside.rutgers.edu'
port = 25
server = smtplib.SMTP(smtp_server, port)
server.starttls()

# Login to the server

# Create the email message
sender_email = os.getenv("SENDER_EMAIL")
receiver_email = 'steamuservni@gmail.com'
message = MIMEMultipart()
message['From'] = sender_email
message['To'] = receiver_email
message['Subject'] = 'Test Email'
body = 'This is a test email sent from Python.'
message.attach(MIMEText(body, 'plain'))

# Send the email
server.sendmail(sender_email, receiver_email, message.as_string())

# Close the connection to the SMTP server
server.quit()