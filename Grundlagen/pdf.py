import PyPDF2

f = open('../TextFiles/schachnovelle.pdf', 'rb')
pdf_reader = PyPDF2.PdfFileReader(f)

print(pdf_reader.numPages)