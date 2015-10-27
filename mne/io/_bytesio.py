# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from io import BytesIO


class PersistentBytesIO(BytesIO):

    def close(self):
        if not self.closed:
            self.value = self.getvalue()
            BytesIO.close(self)
