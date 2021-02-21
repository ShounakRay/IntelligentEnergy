# @Author: Shounak Ray <Ray>
# @Date:   21-Feb-2021 16:02:34:349  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: run_remote.py
# @Last modified by:   Ray
# @Last modified time: 21-Feb-2021 16:02:19:197  GMT-0700
# @License: [Private IP]

import os

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

URL = 'https://codeshare.io/5DeMgr'


# Initialize Driver
chrome_options = Options()
chrome_options.add_argument("--start-maximized")
# chrome_options.add_argument("--kiosk")
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")

driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

driver.get(URL)
driver.find_element_by_id('codeshare-download-button').click()
list(os.path)

# EOF

# EOF
