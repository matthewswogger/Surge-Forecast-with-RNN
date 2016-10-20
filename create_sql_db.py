import numpy as np
import pandas as pd
import sqlite3

# Create a connection and create the db
conn = sqlite3.connect('surge_data.db')

# Create the cursor for use
c = conn.cursor()

# Create table
c.execute('''CREATE TABLE surge (date timestamp, point real, surge real, eta real)''')

# Insert all data into sql db
for i in np.array(pd.read_csv('mydata.csv')):
    sql_insert = "INSERT INTO surge VALUES ('{}',{},{},{})".format(i[0],i[1],i[2],i[3])
    c.execute(sql_insert)

# Save (commit) the changes
conn.commit()

# Close the connection to sql db
# Just be sure any changes have been committed or they will be lost.
conn.close()
