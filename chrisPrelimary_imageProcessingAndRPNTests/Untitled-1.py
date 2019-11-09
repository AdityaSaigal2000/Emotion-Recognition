stringy = "\\\"def printThing(stringyIn):\nif stringyIn == stringy[8]:\nreturn stringy[0]+stringy[8]\nelse:\nreturn stringyIn\nprinty = \"stringy = \" + stringy[8]for i in stringy:\nprinty += printThing(i)\nprinty += stringy[8]\nfor i in stringy:\nprinty += i\nprint (printy)"
stringy = "def printThing(stringyIn):
    if stringyIn == stringy[28]:
        return stringy[0]+stringy[9]
    elif stringyIn == stringy[1]:
        return stringy[0]+stringy[1]
    elif stringyIn == stringy[0]:
        return stringy[0]+stringy[0]
    else:    
        return stringyIn

printy = "stringy = " + stringy[1]
for i in stringy:
    printy += printThing(i)
printy += stringy[1]
for i in stringy:
    printy += i

print (printy)"

def printThing(stringyIn):
    if stringyIn == stringy[28]:
        return stringy[0]+stringy[9]
    elif stringyIn == stringy[1]:
        return stringy[0]+stringy[1]
    elif stringyIn == stringy[0]:
        return stringy[0]+stringy[0]
    else:    
        return stringyIn

printy = "stringy = " + stringy[1]
for i in stringy:
    printy += printThing(i)
printy += stringy[1]
for i in stringy:
    printy += i

print (printy)

