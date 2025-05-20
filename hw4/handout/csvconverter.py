import csv
from io import StringIO

data = """Time Entered,Time Left,Person ID,Group of People,Food Item,Drink,Size,Notes
8:56,9:15,1,F,T,F,h,v,"Group of coworkers ordered hot drinks."
8:58,9:25,2,F,T,F,i,t,"Person bought an iced coffee."
9:00,9:20,3,F,F,T,r,g,"Person enjoyed a refreshing drink."
9:05,9:35,4,T,F,T,h,t,"Group of friends ordered hot beverages."
9:10,9:30,5,F,T,T,i,g,"Person grabbed a grande iced coffee."
9:15,9:40,6,F,T,F,h,g,"Person seemed relaxed, enjoyed a large hot drink."
9:20,9:50,7,T,F,F,i,t,"Group of colleagues ordered iced beverages."
9:25,9:45,8,F,T,F,r,v,"Person bought a venti-sized refresher."
9:30,9:55,9,F,T,T,i,t,"Person bought an iced coffee."
9:35,10:00,10,T,F,F,h,g,"Group of students ordered hot drinks."
9:40,10:05,11,F,T,F,i,g,"Person grabbed a grande iced coffee."
9:45,10:15,12,F,F,T,r,t,"Person bought a tall-sized refresher."
9:50,10:20,13,F,T,T,i,v,"Person bought a venti iced coffee."
9:55,10:25,14,T,F,T,h,t,"Group of friends ordered hot beverages."
10:00,10:30,15,F,T,F,h,g,"Person seemed relaxed, enjoyed a large hot drink."
10:05,10:35,16,T,F,F,i,t,"Group of colleagues ordered iced beverages."
10:10,10:40,17,F,T,T,i,g,"Person bought a grande iced coffee."
10:15,10:45,18,F,F,T,r,g,"Person bought a grande-sized refresher."
10:20,10:50,19,F,T,F,i,t,"Person bought an iced coffee."
10:25,10:55,20,T,F,F,h,v,"Group of coworkers ordered hot drinks."
10:30,10:56,21,F,T,F,i,g,"Person grabbed a grande iced coffee."
10:35,10:56,22,F,F,T,r,t,"Person bought a tall-sized refresher."
10:40,10:56,23,F,T,T,i,v,"Person bought a venti iced coffee."
10:45,10:56,24,T,F,T,h,t,"Group of friends ordered hot beverages."
10:50,10:56,25,F,T,F,h,g,"Person seemed relaxed, enjoyed a large hot drink."
10:55,10:56,26,T,F,F,i,t,"Group of colleagues ordered iced beverages."
11:00,11:00,27,F,T,T,i,g,"Person bought a grande iced coffee."
11:00,11:00,28,T,F,F,h,v,"Group of coworkers ordered hot drinks."
11:00,11:00,29,F,F,T,r,t,"Person bought a tall-sized refresher."
11:00,11:00,30,F,T,F,i,t,"Person bought an iced coffee."
11:00,11:00,31,T,F,F,h,t,"Group of friends ordered hot beverages."
11:00,11:00,32,F,T,F,h,g,"Person seemed relaxed, enjoyed a large hot drink."
11:00,11:00,33,T,F,F,i,t,"Group of colleagues ordered iced beverages."
11:00,11:00,34,F,T,T,i,v,"Person bought a venti iced coffee."
11:00,11:00,35,T,F,T,h,t,"Group of friends ordered hot beverages."
11:00,11:00,36,F,T,F,h,g,"Person seemed relaxed, enjoyed a large hot drink."
11:00,11:00,37,T,F,F,i,t,"Group of colleagues ordered iced beverages."
11:00,11:00,38,F,T,T,i,g,"Person bought a grande iced coffee."
11:00,11:00,39,F,F,T,r,g,"Person bought a grande-sized refresher."
11:00,11:00,40,F,T,F,i,t,"Person bought an iced coffee."
11:00,11:00,41,T,F,F,h,v,"Group of coworkers ordered hot drinks."
11:00,11:00,42,F,F,T,r,t,"Person bought a tall-sized refresher."
11:00,11:00,43,F,T,T,i,v,"Person bought a venti iced coffee."
11:00,11:00,44,T,F,T,h,t,"Group of friends ordered hot beverages."
11:00,11:00,45,F,T,F,h,g,"Person seemed relaxed, enjoyed a large hot drink."
11:00,11:00,46,T,F,F,i,t,"Group of colleagues ordered iced beverages."
11:00,11:00,47,F,T,T,i,g,"Person bought a grande iced coffee."
11:00,11:00,48,F,F,T,r,g,"Person bought a grande-sized refresher."
11:00,11:00,49,F,T,F,i,t,"Person bought an iced coffee."
11:00,11:00,50,T,F,F,h,t,"Group of friends ordered hot beverages."
11:00,11:00,51,F,T,F,h,g,"Person seemed relaxed, enjoyed a large hot drink."
11:00,11:00,52,T,F,F,i,t,"Group of colleagues ordered iced beverages."
11:00,11:00,53,F,T,T,i,v,"Person bought a venti iced coffee."
11:00,11:00,54,T,F,T,h,t,"Group of friends ordered hot beverages."
11:00,11:00,55,F,T,F,h,g,"Person seemed relaxed, enjoyed a large hot drink."
11:00,11:00,56,T,F,F,i,t,"Group of colleagues ordered iced beverages."
11:00,11:00,57,F,T,T,i,g,"Person bought a grande iced coffee."
11:00,11:00,58,F,F,T,r,g,"Person bought a grande-sized refresher."
11:00,11:00,59,F,T,F,i,t,"Person bought an iced coffee."
11:00,11:00,60,T,F,F,h,v,"Group of coworkers ordered hot drinks."
11:00,11:00,61,F,F,T,r,t,"Person bought a tall-sized refresher."
11:00,11:00,62,F,T,T,i,v,"Person bought a venti iced coffee."
11:00,11:00,63,T,F,T,h,t,"Group of friends ordered hot beverages."
11:00,11:00,64,F,T,F,h,g,"Person seemed relaxed, enjoyed a large hot drink."
11:00,11:00,65,T,F,F,i,t,"Group of colleagues ordered iced beverages."""

csv_file = StringIO(data)

# Reading the CSV data into dictionaries
reader = csv.DictReader(csv_file)

# Convert the reader to a list of dictionaries
data_list = list(reader)


def write_csv(data, filename):
    # Extract fieldnames from the first item. This assumes all items have the same keys.
    fieldnames = data[0].keys()

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)

# Call the function with your data and desired filename
write_csv(data_list, 'output.csv')