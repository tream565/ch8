# ch8
從路徑中的檔案尋找特徵
```
def get_string_features(path,hasher):
   
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
利用sklearn對特徵進行判斷此程式是否為惡意程式
```
def scan_file(path):
    if not os.path.exists("saved_detector.pkl"):
        print "It appears you haven't trained a detector yet!  Do this before scanning files."
        sys.exit(1)
    with open("saved_detector.pkl") as saved_detector:
        classifier, hasher = pickle.load(saved_detector)
    features = get_string_features(path,hasher)
    result_proba = classifier.predict_proba([features])[:,1]

    if result_proba > 0.5:
        print "It appears this file is malicious!",`result_proba`
    else:
        print "It appears this file is benign.",`result_proba`
```
利用隨機樹做訓練
```
def train_detector(benign_path,malicious_path,hasher):
    def get_training_paths(directory):
        targets = []
        for path in os.listdir(directory):
            targets.append(os.path.join(directory,path))
        return targets
    malicious_paths = get_training_paths(malicious_path)
    benign_paths = get_training_paths(benign_path)
    X = [get_string_features(path,hasher) for path in malicious_paths + benign_paths]
    y = [1 for i in range(len(malicious_paths))] + [0 for i in range(len(benign_paths))]
    classifier = tree.RandomForestClassifier(64)
    classifier.fit(X,y)
    pickle.dump((classifier,hasher),open("saved_detector.pkl","w+"))
```
評估模型
```
def cv_evaluate(X,y,hasher):
    import random
    from sklearn import metrics
    from matplotlib import pyplot
    from sklearn.cross_validation import KFold
    X, y = numpy.array(X), numpy.array(y)
    fold_counter = 0
    for train, test in KFold(len(X),2,shuffle=True):
        training_X, training_y = X[train], y[train]
        test_X, test_y = X[test], y[test]
        classifier = RandomForestClassifier(64)
        classifier.fit(training_X,training_y)
        scores = classifier.predict_proba(test_X)[:,-1]
        fpr, tpr, thresholds = metrics.roc_curve(test_y, scores)
        pyplot.semilogx(fpr,tpr,label="ROC curve".format(fold_counter))
        fold_counter += 1
        break
    pyplot.xlabel("detector false positive rate")
    pyplot.ylabel("detector true positive rate")
    pyplot.title("Detector ROC curve")
    pyplot.legend()
    pyplot.grid()
    pyplot.show()
```
![曲線](https://github.com/tream565/ch8/blob/main/ch8.png)
