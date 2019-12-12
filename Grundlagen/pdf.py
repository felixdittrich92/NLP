import PyPDF2

# binary lesen und dann die pdf einlesen
f = open('../TextFiles/schachnovelle.pdf', 'rb')
pdf_reader = PyPDF2.PdfFileReader(f)

print(pdf_reader.numPages)

# erste Seite Text anzeigen
page_one = pdf_reader.getPage(0)
page_one_text = page_one.extractText()
print(page_one_text)
f.close()

# Seite aus PDF in neues Dokument "kopieren"
f = open('../TextFiles/schachnovelle.pdf', 'rb')
pdf_reader = PyPDF2.PdfFileReader(f)

first_page = pdf_reader.getPage(0)

pdf_writer = PyPDF2.PdfFileWriter()
pdf_writer.addPage(first_page)
pdf_output = open('Neues_Dokument.pdf', 'wb')
pdf_writer.write(pdf_output)
pdf_output.close()
f.close()

# jede Seite als String in Liste
f = open('../TextFiles/schachnovelle.pdf', 'rb')

pdf_text = list()
pdf_reader = PyPDF2.PdfFileReader(f)

for p in range(pdf_reader.numPages):
    page = pdf_reader.getPage(p)

    pdf_text.append(page.extractText())

f.close()
print(pdf_text)