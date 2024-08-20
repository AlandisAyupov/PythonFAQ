import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv() 


# Connect to the SMTP server
smtp_server = 'smtp.gmail.com'
port = 587
server = smtplib.SMTP(smtp_server, port)
server.starttls()

# Login to the server
server.login(os.getenv("EMAIL"), os.getenv("PASS"))

# Create the email message
sender_email = os.getenv("EMAIL")
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