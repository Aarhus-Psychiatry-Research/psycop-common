from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

#Randomisering til train og test set ud fra patient
train_inds, test_inds = next(GroupShuffleSplit(random_state=42, test_size=0.3).split(b_gradueret, groups=b_gradueret["kCpr"]))
b_train, b_test = b_gradueret.iloc[train_inds], b_gradueret.iloc[test_inds]

