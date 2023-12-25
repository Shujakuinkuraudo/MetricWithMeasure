import AllInOne
import AllInOne.Train 
from AllInOne.utils import seed_everything
seed_everything(42)
a = AllInOne.Train.Train()
a.train_all()