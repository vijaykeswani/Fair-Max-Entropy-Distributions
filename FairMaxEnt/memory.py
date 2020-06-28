

class Memory(object):
    # dummy memory object that does not store data
    def __init__(self):
        super(Memory, self).__init__()
        pass
    def __contains__(self, item):
        return False
    def __getitem__(self, item):
        return 0
    def __setitem__(self, key, value):
        pass
    def get(self, key, replacementValue):
        """
            if item is not in the memory, return replacementValue otherwise
            return stored value
        """
        return replacementValue

    def reset(self):
        pass

class MemoryHash(dict):
    def reset(self):
        self.clear()


class MemoryTrie(Memory):
    def __init__(self):
        super(MemoryTrie, self).__init__()
        self.value = 0
        self.children = []

    def __contains__(self, item):
        if len(item) == 0:
            return True
        index = item[0]
        if index >= len(self.children):
            return False
        child = self.children[index]
        if child is None:
            return False
        return item[1:] in child

    def __getitem__(self, item):
        if len(item) == 0:
            return self.value
        index = item[0]
        if index >= len(self.children):
            return 0
        child = self.children[index]
        if child is None:
            return 0
        return child[item[1:]]

    def __setitem__(self, key, value):
        if len(key) == 0:
            return self.value
        index = key[0]
        if index >= len(self.children):
            self.children.extend([None]*(index+1-len(self.children)))
        child = self.children[index]
        if child is None:
            child = self.children[index] = MemoryTrie()
        if len(key) == 1:
            child.value = value
        else:
            child[key[1:]] = value


    def get(self, key, replacementValue):
        if key in self:
            return self[key]
        return replacementValue

    def reset(self):
        self.value = 0
        self.children = []

