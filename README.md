# ch8
從路徑豬的檔案尋找特徵
```
def get_string_features(path,hasher):
    # extract strings from binary file using regular expressions
    chars = r" -~"
    min_length = 5
    string_regexp = '[%s]{%d,}' % (chars, min_length)
    file_object = open(path)
    data = file_object.read()
    pattern = re.compile(string_regexp)
    strings = pattern.findall(data)


    string_features = {}
    for string in strings:
        string_features[string] = 1


    hashed_features = hasher.transform([string_features])


    hashed_features = hashed_features.todense()
    hashed_features = numpy.asarray(hashed_features)
    hashed_features = hashed_features[0]


    print "Extracted {0} strings from {1}".format(len(string_features),path)
    return hashed_features
```
