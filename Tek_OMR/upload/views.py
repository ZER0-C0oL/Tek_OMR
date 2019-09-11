from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
from .img_process import process_files
from .find_marks import find_marks

def simple_upload(request):
    if request.method == 'POST' and request.FILES['image']:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save( 'img/' + image.name,image)
        scan_path = BASE_DIR + '/media/' + filename
        uploaded_file_url_image = fs.url(filename)
        svg = request.FILES['svg']
        fs = FileSystemStorage()
        filename = fs.save('svg/' + svg.name, svg)
        uploaded_file_url_svg = fs.url(filename)
        template_path = BASE_DIR + '/media/' + filename
        json = request.FILES['json']
        fs = FileSystemStorage()
        valid_ans = process_files(TEMPLATE_PATH=template_path, SCAN_PATH=scan_path)
        filename = fs.save('json/' + json.name, json)
        json_path = BASE_DIR + '/media/' + filename
        marks = find_marks(valid_ans, json_path)
        uploaded_file_url_json = fs.url(filename)
        return render(request, 'upload/imageform_form.html', {
            'uploaded_file_url_image': uploaded_file_url_image,
            'uploaded_file_url_svg': uploaded_file_url_svg,
            'uploaded_file_url_json': uploaded_file_url_json,
            'answer': str(marks) + ' / 100'
        })
    return render(request, 'upload/imageform_form.html')


def home(request):
    return render(request, 'home.html')
