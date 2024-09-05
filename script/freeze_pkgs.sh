## output the current pkgs with version number
pip3 freeze > requirements.txt
## download the pkgs based on requirements
# pip download -d DIR -r requirements.txt
pip wheel -w DIR -r requirements.txt
## install the required pkgs offline
pip3 install --no-index --find-links=DIR -r requirements.txt
## install the required pkgs onlin
pip3 install -r requirements.txt