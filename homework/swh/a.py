class Person:

    def __init__(self, name):
        self.name = name

    def show(self):

        print(
            "{} loves mathematics. I do hope this training will be a complete success!".format(
                self.name
            )
        )


p1 = Person("Peter")
p1.show()
