pip3 freeze > requirements.txt   # save the current pkgs
# pip download -d DIR -r requirements.txt
pip wheel -w DIR -r requirements.txt  # download all the pkgs
pip3 install --no-index --find-links=DIR -r requirements.txt  # install all the pkgs offline
pip3 install -r requirements.txt # install all the pkgs online