import cv2, time, pandas
from datetime import datetime

first_frame = None
status_list = [None, None]
times =[]
#pandas dataframe with 2 columns
df = pandas.DataFrame(columns=["Start", "End"])

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    #first frame is 0, and empty
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     #removes noice and makes picture blurry
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        #assigning a graye numpy to first frame
        first_frame = gray
        #after continue we crab second frame
        continue

    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    #the bigger the nr on iterations the clearer image
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
    #finding all the contours for images and it will be stored in cnts variable
    contours, hierarchy = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        #looking for an object which is bigger than 10000px for the green frame
        if cv2.contourArea(contour) < 10000:
            continue
        #status from 0 to 1 when there is object in the camera and 1 to 0 when it exits the camera
        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    #for recording times of the statuses if something appears to the camera and disapears
    status_list.append(status)

    status_list = status_list[-2:] # only to keep last 2 statuses of the video frame, so in long videos its not eating up the memory

    #last 2 item of the status list = "status_list[0,1]", last item is -1 and "status_list" with item -2 is 0, then we want to record the datetime of these event in a list times = []
    if status_list[-1] == 1 and status_list[-2] == 0: # when it changes from 0 to 1
        #recording to times = []
        times.append(datetime.now())
    #when "status_list[1,0]" changes from 1 to 0
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

print(status_list)
print(times)

#adding times to data frame in pandas based on start and end time on the same row and different columns
for i in range(0, len(times), 2):
    df = df.append({"Srart":times[i], "End":times[i + 1]}, ignore_index = True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows
