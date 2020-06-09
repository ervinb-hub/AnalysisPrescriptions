import requests
import re
import json
from bs4 import BeautifulSoup


def get_soup_object(url):
    # set a user-agent to be sent with request
    headers = {
        "user-agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) \
        Chrome/61.0.3163.100 Safari/537.36"
    }

    print("Retrieving from: " + url)

    response  = requests.get(url, headers)
    if (response.ok):
        # parse the raw HTML into a `soup' object
        soupObj = BeautifulSoup(response.text, "html.parser")

        return soupObj
    else:
        print("There was a problem reading from the URL: " + url)
        return None




def parse_n_save():
	bsObj = get_soup_object('https://openprescribing.net/bnf/')
	data = bsObj.find_all('a')
	pattern = re.compile('/bnf/[0-9]{2,6}')

	results = []

	for i in data:
		temp_dict = {}
		href = i.attrs['href']
		match = re.search(pattern, href)
		if match:
			try:
				code = href.split('/')[2]
				desc = i.text.split(':')[1].strip()
				re.sub(r'[^a-zA-Z0-9_]', '', desc)
				temp_dict['code'] = code
				temp_dict['desc'] = desc
				results.append(temp_dict)
			except Exception as e:
				print(str(e))

	with open('sections.json', 'w') as output_file:
		json.dump(results, output_file)
		print('Results written to sections.json')

