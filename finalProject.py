#import needed libraries
import cv2
import face_recognition

#faces and encodings
NadalFace = face_recognition.load_image_file('/home/sabrina/project/data/rafael_nadal.jpg')
NadalEncode = face_recognition.face_encodings(NadalFace)[0]

RogerFace = face_recognition.load_image_file('/home/sabrina/project/data/roger_federer.jpg')
RogerEncode = face_recognition.face_encodings(RogerFace)[0]

NovakFace = face_recognition.load_image_file('/home/sabrina/project/data/novak.jpg')
NovakEncode = face_recognition.face_encodings(NovakFace)[0]

Encodings = [NadalEncode, RogerEncode, NovakEncode]
Names = ['rafael nadal', 'roger federer', 'novak djokovic']

#create testing image
font = cv2.FONT_HERSHEY_SIMPLEX
test = face_recognition.load_image_file('/home/sabrina/project/data/three2.jpg')
positions = face_recognition.face_locations(test)
allEncodings = face_recognition.face_encodings(test, positions)

test = cv2.cvtColor(testImage, cv2.COLOR_RGB2BGR)

#face recognition for loop + showing the window using cv2
for (top, right, bottom, left), face_encoding in zip(positions, allEncodings):
    matches= face_recognition.compare_faces(Encodings, face_encoding)
    if True in matches:
        first_match_index = matches.index(True)
        name = Names[first_match_index]
    cv2.rectangle(test, (left, top), (right, bottom), (252, 232, 131), 2)
    cv2.putText(test, name, (left, top-6), font, .75, (252, 131, 202), 1)
cv2.imshow("result", test)
cv2.moveWindow('myWindow', 0, 0)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
