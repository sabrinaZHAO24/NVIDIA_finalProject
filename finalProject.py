import cv2
import face_recognition

NadalFace = face_recognition.load_image_file('/home/sabrina/project/data/rafael_nadal.jpg')
NadalEncode = face_recognition.face_encodings(NadalFace)[0]

RogerFace = face_recognition.load_image_file('/home/sabrina/project/data/roger_federer.jpg')
RogerEncode = face_recognition.face_encodings(RogerFace)[0]

NovakFace = face_recognition.load_image_file('/home/sabrina/project/data/novak.jpg')
NovakEncode = face_recognition.face_encodings(NovakFace)[0]

Encodings = [NadalEncode, RogerEncode, NovakEncode]
Names = ['rafael nadal', 'roger federer', 'novak djokovic']

font = cv2.FONT_HERSHEY_SIMPLEX
testImage = face_recognition.load_image_file('/home/sabrina/project/data/three2.jpg')
positions = face_recognition.face_locations(testImage)
allEncodings = face_recognition.face_encodings(testImage, positions)

testImage = cv2.cvtColor(testImage, cv2.COLOR_RGB2BGR)

for (top, right, bottom, left), face_encoding in zip(positions, allEncodings):
    matches= face_recognition.compare_faces(Encodings, face_encoding)
    if True in matches:
        first_match_index = matches.index(True)
        name = Names[first_match_index]
    cv2.rectangle(testImage, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(testImage, name, (left, top-6), font, .75, (0, 255, 255), 1)
cv2.imshow("result", testImage)
cv2.moveWindow('myWindow', 0, 0)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
