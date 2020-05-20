from flask import Flask, jsonify, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/downloads/'
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__, static_url_path="/static")
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# limit upload size upto 8mb
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def deflt():
   images = os.listdir(os.path.join(app.static_folder, "images"))
   return render_template('index.html', images=images)
def get_coordinates_text(in_txt):
    if in_txt == 'Bisceglie':
        return('40.76203918457031, 15.639488220214844','42.39269256591797, 14.403570175170898//@42.39269256591797, 14.403570175170898','42.39269256591797, 14.403570175170898', '6')
    elif in_txt == 'Escauduvres':
        return('50.30093765258789, 3.303251028060913','50.80487823486328, 1.917317509651184//@50.80487823486328, 1.917317509651184','50.80487823486328, 1.917317509651184', '1')
    elif in_txt == 'Golden Bridge':
        return('15.710230827331543, 104.87944030761719','15.710230827331543, 104.87944030761719//@15.710230827331543, 104.87944030761719','15.710230827331543, 104.87944030761719', '0')
    elif in_txt == 'Oriental':
        return('37.65284729003906, -2.735166311264038','30.91046905517578, 6.427838325500488//@30.91046905517578, 6.427838325500488','30.91046905517578, 6.427838325500488', '11')
    elif in_txt == 'Tvaro':
        return('49.17348861694336, 16.625314712524414','50.048912048339844, 15.335355758666992//@50.048912048339844, 15.335355758666992','50.048912048339844, 15.335355758666992', '5')
    elif in_txt == 'Sasebo':
        return('34.686180114746094, 130.4123077392578','34.686180114746094, 130.4123077392578//@34.686180114746094, 130.4123077392578','34.686180114746094, 130.4123077392578', '0')
    elif in_txt == 'Lake Ontario':
        return('43.2764778137207, -78.41609191894531','47.115169525146484, -83.9014663696289//@47.115169525146484, -83.9014663696289','47.115169525146484, -83.9014663696289','21')
    elif in_txt == 'Guldental':
        return('50.16755676269531, 8.395276069641113','50.09747314453125, 6.494518756866455//@50.09747314453125, 6.494518756866455','50.09747314453125, 6.494518756866455', '5')
    elif in_txt == 'Sirmione':
        return('45.313232421875, 10.036974906921387','45.313232421875, 10.036974906921387//@45.313232421875, 10.036974906921387','45.313232421875, 10.036974906921387', '0')
    elif in_txt == 'Pingshan District':
        return('42.774200439453125, 133.25958251953125','36.12027359008789, 117.1050033569336//@36.12027359008789, 117.1050033569336','36.12027359008789, 117.1050033569336', '8')
    else:
        return('125.5,78.9','55.5,79.9//@55.5,79.9','', '')

def get_coordinates_image(in_img):
    if in_img == 'Q691876_0.jpg':
        return('47.033233642578125, 15.794718742370605','47.033233642578125,  15.794718742370605//@47.033233642578125,  15.794718742370605','47.033233642578125,  15.794718742370605','0')
    elif in_img == 'Q23178_0.jpg':
        return('40.76203918457031, 15.639488220214844','45.1514892578125, 11.642662048339844//@45.1514892578125, 11.642662048339844','45.1514892578125, 11.642662048339844','3')
    elif in_img == 'Q630749_0.png':
        return('50.30093765258789, 3.303251028060913','48.22926330566406,  6.837433338165283//@48.22926330566406,  6.837433338165283','48.22926330566406,  6.837433338165283','4')
    elif in_img == 'Q1789630_0.jpg':
        return('49.17348861694336, 16.625314712524414','49.56821060180664, 17.874563217163086//@49.56821060180664, 17.874563217163086','49.56821060180664, 17.874563217163086','5')
    elif in_img == 'Q680361_2.png':
        return('50.16755676269531, 8.395276069641113','47.569766998291016, 9.26429271697998//@47.569766998291016, 9.26429271697998','47.569766998291016, 9.26429271697998','6')
    elif in_img == 'Q630749_1.png':
        return('50.30093765258789, 3.303251028060913','49.481712341308594, 5.578824043273926//@49.481712341308594, 5.578824043273926','49.481712341308594, 5.578824043273926','7')
    elif in_img == 'Q680361_1.jpg':
        return('50.16755676269531, 8.395276069641113','50.096519470214844, 7.314050197601318//@50.096519470214844, 7.314050197601318','50.096519470214844, 7.314050197601318','9')
    elif in_img == 'Q55954790_1.jpg':
        return('15.710230827331543,  104.87944030761719','11.835397720336914, -8.977499961853027//@11.835397720336914, -8.977499961853027','11.835397720336914, -8.977499961853027','14')
    elif in_img == 'Q1789630_2.png':
        return('49.17348861694336,  16.625314712524414','52.757442474365234,  16.558185577392578//@52.757442474365234,  16.558185577392578','52.757442474365234,  16.558185577392578','16')
    elif in_img == 'Q112019_0.jpg':
        return('45.313232421875, 10.036974906921387','45.313232421875, 10.036974906921387//@45.313232421875, 10.036974906921387','45.313232421875, 10.036974906921387','0')
    elif in_img == 'Q680361_0.jpg':
        return('50.16755676269531, 8.395276069641113','50.096519470214844, 7.314050197601318//@50.096519470214844, 7.314050197601318', '50.096519470214844, 7.314050197601318', '10')
    elif in_img == 'Q23178_0.jpg':
        return('40.76203918457031, 15.639488220214844','45.1514892578125,11.642662048339844','45.1514892578125,11.642662048339844//@45.1514892578125,11.642662048339844','45.1514892578125,11.642662048339844','3')
    else:
        return('15.710230827331543, 104.87944030761719','15.710230827331543, 104.87944030761719//@15.710230827331543, 104.87944030761719','15.710230827331543, 104.87944030761719', '0')



@app.route('/text-coordinate',methods = ['POST'])
def resulttext():
    if request.method == 'POST':
        in_txt = request.form['inTxt']
        saddr, daddr, pred, rank = get_coordinates_text(in_txt)
        images = os.listdir(os.path.join(app.static_folder, "images"))
        return render_template('index.html', saddr=saddr, daddr=daddr, images=images, pred=pred, rank=rank)

@app.route('/image-coordinate',methods = ['POST'])
def resultimage():
    if request.method == 'POST':
        in_img = request.form['inImg']
        saddr, daddr, pred, rank = get_coordinates_image(in_img)
        images = os.listdir(os.path.join(app.static_folder, "images"))
        return render_template('index.html', saddr=saddr, daddr=daddr, images=images, pred=pred, rank=rank)
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=6060)