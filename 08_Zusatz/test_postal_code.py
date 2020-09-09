# -*- coding: utf-8 -*-
# pip3 install address
from address import AddressParser
import textract

text = textract.process('/home/felix/Desktop/NLP/08_Zusatz/gray_300dpi.jpg', encoding='ascii')
text = text.strip()
print(text)

ap = AddressParser()
address = ap.parse_address(str(text.strip()))
#print "Address is: {0} {1} {2} {3}".format(address.house_number, address.street_prefix, address.street, address.street_suffix)

#print(str(address))
