## Iban Berganzo-Besga ## 
### Post-doctoral Fellow
#### Ramsey Laboratory for Environmental Archaeology (RLEA)​
#### University of Toronto Mississauga (UTM)​
### Associate Researcher​
#### Landscape Archaeology Research Group (GIAP)​
#### Catalan Institute of Classical Archaeology (ICAC)​

import os
import json
import argparse
import cv2
img_type = '.jpg'
annot_type = '.txt'
shap_name = 'polyline'

def via_to_yolo(folder_path, outFolder):
    for filename_in in os.listdir(folder_path):
        if filename_in.endswith(('.json', '.JSON')):
            json_file = os.path.join(folder_path, filename_in)
            print(json_file)
            with open(json_file) as f:
                data = json.load(f)



            all_points_x = None
            all_points_y = None 
            filenames = []

            outFolder = outFolder + '/'+filename_in.replace('.json','')
            if not os.path.exists(outFolder):
                os.makedirs(outFolder)

            for image_id, metadata in data['_via_img_metadata'].items():
                image_name = str(metadata['filename']).replace(img_type,'')
                regions = metadata['regions']
                if(not len(regions)):
                    continue

                imgPath = folder_path + '/'+filename_in.replace('.json','')+'/' + image_name + img_type
                img = cv2.imread(imgPath)
                if img is None:
                    continue
                height, width, c = img.shape
                
                filename_out = image_name + annot_type
                print(outFolder + '/'+ filename_out)
                for region in regions:
                    shape_attributes = region.get('shape_attributes', {})
                    if shape_attributes.get('name') == shap_name:
                        x_values = shape_attributes.get('all_points_x', [])
                        y_values = shape_attributes.get('all_points_y', [])
                        all_points_x = x_values
                        all_points_y = y_values
                        
                        values = []
                        values.append(0) # only one class
                        for i in range(len(all_points_x)):
                            values.append(round(all_points_x[i]/width, 4))
                            values.append(round(all_points_y[i]/height, 4))
                        
                        outpath = os.path.join(outFolder, filename_out)
                        result = ' '.join(map(str, values))
                        if not outpath.endswith('.txt'):
                            outpath += '.txt'
                        with open(outpath, 'a' if os.path.exists(outpath) else 'w') as file1:
                            file1.write(result + '\n')

def main():
    parser = argparse.ArgumentParser(description='VIA to YOLO Format')
    parser.add_argument('--img', nargs=2, metavar=('xSize', 'ySize'), type=int, required=True, help='Image size in pixels horizontally and vertically')   
    # args = parser.parse_args()
    # xSize, ySize = args.img

    tarin_v = '/train0'
    cwd = os.getcwd()+tarin_v  # Get the current working directory (cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    # print("Files in %r: %s" % (cwd, files))

    folder = cwd+'/via/v1'
    outFolder = cwd+'/yolo/v1'
    via_to_yolo(folder, outFolder)

if __name__ == "__main__":
    main()

