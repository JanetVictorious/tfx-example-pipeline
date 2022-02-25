"""Features test file.
"""

import tensorflow as tf

from models import features


class FeaturesTest(tf.test.TestCase):

    def testTransformedNames(self):
        names = ["f1", "cf"]
        self.assertEqual(["f1_xf", "cf_xf"], features.transformed_names(names))


if __name__ == "__main__":
    tf.test.main()
