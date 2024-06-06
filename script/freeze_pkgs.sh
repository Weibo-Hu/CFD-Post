pip3 freeze > requirements.txt
# pip download -d DIR -r requirements.txt
pip wheel -w DIR -r requirements.txt
pip3 install --no-index --find-links=DIR -r requirements.txt