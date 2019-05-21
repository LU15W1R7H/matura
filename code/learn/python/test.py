import math
import utiltest

print('*' * 10)

number = 19
mark = 4.9
is_true = True
print(number)


name = 'luis'
birth_year = 2000
age = 2019 - int(birth_year) # int-function
print(type(birth_year))
print(type(age))

long_string = '''
Hallo mein Name ist luis
ich bin mehrere Zeilen lang.
'''

print(long_string[-4])
print(long_string[0:3])
print(long_string[5:])

copy_of_string = long_string[:]

message = f'{name} was born in [{birth_year}]'
print(message)
print(len(message))
print(message.upper())
print(message.find('w'))
print('luis' in message)

print(math.floor(2.9))

for letter in message:
    print(letter)

some_list = ['Hi', 'this', 'is', 'an', 'element']
some_list.append('BYE')
other_list = range(10)
print(other_list[5])

print('---------------------')

for x in range(3):
    for y in range(3):
        print(x,y)

some_tuple = (2,3)
x,y = some_tuple # also works for lists


# dictionary
customer = {
    'name': 'Luis Wirth',
    'age': 18,
    'gender': 'male'
}
print(customer['name'])

def func():
    print('im a function')

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def move(self):
        print('move')

    def draw(self):
        print('draw')

point1 = Point(3, 4)
point1.draw()
point1.lol = 5 # creating new attribute (!?)

print(utiltest.testFun())
