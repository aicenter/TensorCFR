# Results of Experiments

## Convergence over CFR+ steps

* `domain01`
    * [1,000 steps](https://gitlab.com/beyond-deepstack/TensorCFR/snippets/1710646)
    * [10,000 steps](https://gitlab.com/beyond-deepstack/TensorCFR/snippets/1710645)
* `matching_pennies`
    * [1,000 steps](https://gitlab.com/snippets/1710648)
    * [10,000 steps](https://gitlab.com/snippets/1710649)
    
## Time measurements

* `tensorcfr_on_goofspiel5.py` ~40 seconds
    * run 1
    
        ```
        mathemage@mathemage-XPS-15-9560:~$ gpu-monitor
        mathemage-XPS-15-9560  Mon May  7 10:47:46 2018
        [0] GeForce GTX 1050 | 55'C,   1 % |  3742 /  4042 MB | mathemage(2613M) root(607M) mathemage(128M) mathemage(60M) mathemage(321M)
        mathemage-XPS-15-9560  Mon May  7 10:47:49 2018
        [0] GeForce GTX 1050 | 57'C,   9 % |  3752 /  4042 MB | mathemage(2623M) root(607M) mathemage(128M) mathemage(60M) mathemage(321M)
        mathemage-XPS-15-9560  Mon May  7 10:47:52 2018
        [0] GeForce GTX 1050 | 58'C,  12 % |  3752 /  4042 MB | mathemage(2623M) root(607M) mathemage(128M) mathemage(60M) mathemage(321M)
        mathemage-XPS-15-9560  Mon May  7 10:47:56 2018
        [0] GeForce GTX 1050 | 58'C,  10 % |  3752 /  4042 MB | mathemage(2623M) root(607M) mathemage(128M) mathemage(60M) mathemage(321M)
        mathemage-XPS-15-9560  Mon May  7 10:47:59 2018
        [0] GeForce GTX 1050 | 59'C,  10 % |  3752 /  4042 MB | mathemage(2623M) root(607M) mathemage(128M) mathemage(60M) mathemage(321M)
        mathemage-XPS-15-9560  Mon May  7 10:48:02 2018
        [0] GeForce GTX 1050 | 60'C,  12 % |  3752 /  4042 MB | mathemage(2623M) root(607M) mathemage(128M) mathemage(60M) mathemage(321M)
        mathemage-XPS-15-9560  Mon May  7 10:48:05 2018
        [0] GeForce GTX 1050 | 60'C,  12 % |  3752 /  4042 MB | mathemage(2623M) root(607M) mathemage(128M) mathemage(60M) mathemage(321M)
        mathemage-XPS-15-9560  Mon May  7 10:48:08 2018
        [0] GeForce GTX 1050 | 61'C,  12 % |  3752 /  4042 MB | mathemage(2623M) root(607M) mathemage(128M) mathemage(60M) mathemage(321M)
        mathemage-XPS-15-9560  Mon May  7 10:48:11 2018
        [0] GeForce GTX 1050 | 62'C,  11 % |  3752 /  4042 MB | mathemage(2623M) root(607M) mathemage(128M) mathemage(60M) mathemage(321M)
        mathemage-XPS-15-9560  Mon May  7 10:48:14 2018
        [0] GeForce GTX 1050 | 62'C,  11 % |  3752 /  4042 MB | mathemage(2623M) root(607M) mathemage(128M) mathemage(60M) mathemage(321M)
        mathemage-XPS-15-9560  Mon May  7 10:48:17 2018
        [0] GeForce GTX 1050 | 63'C,  12 % |  3752 /  4042 MB | mathemage(2623M) root(607M) mathemage(128M) mathemage(60M) mathemage(321M)
        mathemage-XPS-15-9560  Mon May  7 10:48:20 2018
        [0] GeForce GTX 1050 | 63'C,  10 % |  3752 /  4042 MB | mathemage(2623M) root(607M) mathemage(128M) mathemage(60M) mathemage(321M)
        mathemage-XPS-15-9560  Mon May  7 10:48:24 2018
        [0] GeForce GTX 1050 | 63'C,  26 % |  1129 /  4042 MB | root(617M) mathemage(128M) mathemage(60M) mathemage(321M)
        mathemage-XPS-15-9560  Mon May  7 10:48:27 2018
        [0] GeForce GTX 1050 | 62'C,   1 % |  1129 /  4042 MB | root(617M) mathemage(128M) mathemage(60M) mathemage(321M)
        ```
        
    * run 2
    
        ```
        mathemage-XPS-15-9560  Mon May  7 11:22:09 2018
        [0] GeForce GTX 1050 | 60'C,   1 % |  3742 /  4042 MB | mathemage(2775M) root(542M) mathemage(127M) mathemage(60M) mathemage(193M) mathemage(32M)
        mathemage-XPS-15-9560  Mon May  7 11:22:12 2018
        [0] GeForce GTX 1050 | 61'C,  12 % |  3752 /  4042 MB | mathemage(2785M) root(542M) mathemage(127M) mathemage(60M) mathemage(193M) mathemage(32M)
        mathemage-XPS-15-9560  Mon May  7 11:22:15 2018
        [0] GeForce GTX 1050 | 61'C,  14 % |  3737 /  4042 MB | mathemage(2785M) root(536M) mathemage(117M) mathemage(60M) mathemage(193M) mathemage(32M)
        mathemage-XPS-15-9560  Mon May  7 11:22:19 2018
        [0] GeForce GTX 1050 | 62'C,  13 % |  3737 /  4042 MB | mathemage(2785M) root(536M) mathemage(117M) mathemage(60M) mathemage(193M) mathemage(32M)
        mathemage-XPS-15-9560  Mon May  7 11:22:22 2018
        [0] GeForce GTX 1050 | 62'C,  13 % |  3737 /  4042 MB | mathemage(2785M) root(536M) mathemage(117M) mathemage(60M) mathemage(193M) mathemage(32M)
        mathemage-XPS-15-9560  Mon May  7 11:22:25 2018
        [0] GeForce GTX 1050 | 63'C,  11 % |  3737 /  4042 MB | mathemage(2785M) root(536M) mathemage(117M) mathemage(60M) mathemage(193M) mathemage(32M)
        mathemage-XPS-15-9560  Mon May  7 11:22:28 2018
        [0] GeForce GTX 1050 | 63'C,  10 % |  3740 /  4042 MB | mathemage(2785M) root(538M) mathemage(119M) mathemage(60M) mathemage(193M) mathemage(32M)
        mathemage-XPS-15-9560  Mon May  7 11:22:31 2018
        [0] GeForce GTX 1050 | 64'C,  13 % |  3731 /  4042 MB | mathemage(2785M) root(530M) mathemage(118M) mathemage(60M) mathemage(193M) mathemage(32M)
        mathemage-XPS-15-9560  Mon May  7 11:22:34 2018
        [0] GeForce GTX 1050 | 65'C,  13 % |  3731 /  4042 MB | mathemage(2785M) root(530M) mathemage(118M) mathemage(60M) mathemage(193M) mathemage(32M)
        mathemage-XPS-15-9560  Mon May  7 11:22:37 2018
        [0] GeForce GTX 1050 | 65'C,  22 % |  3732 /  4042 MB | mathemage(2785M) root(531M) mathemage(118M) mathemage(60M) mathemage(193M) mathemage(32M)
        mathemage-XPS-15-9560  Mon May  7 11:22:40 2018
        [0] GeForce GTX 1050 | 66'C,  30 % |  3731 /  4042 MB | mathemage(2785M) root(531M) mathemage(117M) mathemage(60M) mathemage(193M) mathemage(32M)
        mathemage-XPS-15-9560  Mon May  7 11:22:43 2018
        [0] GeForce GTX 1050 | 66'C,  11 % |  3730 /  4042 MB | mathemage(2785M) root(531M) mathemage(117M) mathemage(60M) mathemage(193M) mathemage(32M)
        mathemage-XPS-15-9560  Mon May  7 11:22:46 2018
        [0] GeForce GTX 1050 | 67'C,   9 % |  3746 /  4042 MB | mathemage(2785M) root(536M) mathemage(127M) mathemage(60M) mathemage(193M) mathemage(32M)
        ```
