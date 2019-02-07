import os
import tensorflow as tf

from src.commons import constants as common_constants
from src.domains.FlattenedDomain import FlattenedDomain
from src.utils.other_utils import activate_script


def get_flattened_oshi_zumo():
    path_to_domain_filename = os.path.join(
        common_constants.PROJECT_ROOT,
        'doc',
        'oshi_zumo',
        'oshizumo__startingCoins_10_locK_3_minBid_1.efg'  # 'oshizumo__startingCoins_9_locK_3_minBid_3.efg'
    )
    return FlattenedDomain.init_from_hkl_file(path_to_domain_filename, domain_name="Oshi-Zumo")


if __name__ == '__main__' and activate_script():
    goofspiel6 = get_flattened_oshi_zumo()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        goofspiel6.print_domain(sess)
