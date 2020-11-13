#define CONV1_WT {-8,-9,6,-38,-20,28,12,31,-14,5,25,5,33,44,-7,-24,24,7,32,3,-5,28,23,27,33,-24,15,7,48,43,-10,-32,-2,26,28,-41,-42,11,44,52,-34,-29,-43,11,56,-47,-37,-26,-9,47,1,-15,-2,49,27,7,-38,-31,-7,12,10,-38,-69,-36,-27,55,15,-10,-9,-5,6,57,52,27,2,11,41,16,38,47,6,6,31,17,52,-49,-8,29,9,19,-63,-54,-59,-32,-42,-26,-39,-61,-54,-29,0,27,13,-50,-11,50,41,-16,-57,-30,18,19,-20,-3,13,-25,32,43,-1,-17,-12,7,21,24,-3,7,12,-25,6,28,7,57,46,28,-17,-1,-10,14,-21,-15,-37,-32,-34,-28,9,-21,5,35,43,33,29,38,2,-28,-57,-12,29,15,-14,-59,-4,-16,18,29,50,-19,4,5,24,41,-11,10,12,14,12,-7,-46,-34,-43,-36,-23,1,5,-18,-34,-32,-9,-41,-30,-31,-38,-26,-39,-5,-58,26,-33,-3,-44,-62}
#define CONV1_WT_SHAPE 200
#define CONV1_BIAS {67,8,14,-2,6,28,58,-1}
#define CONV1_BIAS_SHAPE 8
#define CONV2_WT {-76,-41,38,0,-17,-11,11,-13,-50,-19,34,47,1,-2,-18,16,-24,6,34,85,-5,32,-7,-34,-40,-96,77,-2,40,0,23,53,-23,-37,77,33,11,43,-16,23,-1,-16,123,-5,41,113,10,8,35,-40,73,-30,-25,91,51,23,26,-74,19,-38,9,76,36,-23,63,-53,44,-47,17,78,62,-64,22,-1,-9,54,1,-15,-14,55,-16,19,2,6,-1,-27,-12,25,-14,92,-12,38,17,-8,1,24,32,24,2,20,-37,-82,-41,40,-32,-7,30,-19,-15,-65,-5,82,53,72,-20,22,6,-21,11,-22,-8,18,27,78,-9,-41,-63,12,-41,51,-1,-25,-7,0,-34,74,53,94,-59,-1,3,14,-10,13,-12,2,-125,53,-25,-22,-10,-42,-1,21,20,77,13,43,26,-32,-1,28,19,76,-13,-2,30,-34,-62,32,63,109,-24,-13,-51,24,-62,9,25,82,-74,15,-42,-1,-41,-4,-20,104,-61,6,-43,-36,42,11,59,41,12,-37,-5,18,23,-21,60,80,-42,39,1,19,-9,-71,12,39,-39,69,-18,61,22,-5,-26,-98,83,57,17,-73,-46,-67,63,-77,25,11,-30,-35,-70,-47,32,-68,-23,-4,-48,-12,21,71,32,-2,59,-28,53,-10,-1,24,37,-76,73,31,67,-16,2,-43,84,-72,39,-5,-5,-57,14,-26,86,-5,49,-38,34,-32,20,43,33,-8,-14,-25,11,-8,35,4,25,48,22,41,38,-96}
#define CONV2_WT_SHAPE 288
#define CONV2_BIAS {58,80,51,-110}
#define CONV2_BIAS_SHAPE 4
#define FC1_WT {5,0,-18,-1,4,15,14,9,-2,1,15,9,3,-6,-10,3,9,-26,-15,-15,-7,11,9,21,-9,13,0,0,0,-7,-1,14,14,-15,-8,-14,-8,-3,6,27,4,10,20,-15,14,7,12,0,-5,-11,15,-25,-12,7,21,-2,-5,38,15,5,15,9,6,12,15,-8,-2,-30,2,-9,6,-11,1,1,3,15,5,-13,-7,-5,25,13,-33,13,32,0,8,21,1,-3,17,-8,-14,-15,-28,-50,1,19,-18,3,5,19,22,9,-7,3,4,-4,8,-5,-4,-39,-26,-30,-12,-30,-6,1,8,-2,-9,21,0,1,18,-25,-3,-22,-3,-17,-2,-36,-4,6,-7,-15,34,1,-2,5,20,-44,3,9,5,-33,-5,-16,9,-12,-29,-38,36,5,19,-17,-3,-4,0,-4,-30,21,-47,6,33,12,14,-2,0,8,2,-12,-7,-12,-12,-15,-15,-11,-9,-7,7,-4,11,14,-17,26,-7,-41,-21,21,-19,-31,3,-61,-4,4,2,2,-18,14,6,4,-19,-10,-11,0,-8,-6,-6,-13,-5,-36,4,3,-20,3,7,6,-5,-6,-26,-9,-15,8,-10,-7,-3,-5,12,14,-42,-7,-4,20,26,-7,-30,17,-4,6,-25,41,-18,0,-22,20,-26,-2,2,-7,13,-13,-7,-10,-21,9,-13,-14,-13,-22,-21,1,-11,16,3,16,-4,-2,-26,5,-15,-16,9,-42,-22,6,-10,9,-1,12,-23,6,18,-6,-26,0,-6,4,-17,23,-5,-14,6,7,-2,11,-5,32,-9,2,-22,-10,8,9,-6,-3,9,13,18,-6,32,-30,-6,15,-10,6,10,-13,6,10,-33,2,1,10,-29,29,0,-9,0,5,11,7,6,-14,17,1,9,-22,-3,-9,7,24,-2,5,-8,8,16,-7,-8,-8,3,-4,23,-3,19,21,3,3,11,0,-3,15,10,-22,8,-20,11,-12,-17,0,8,11,7,-26,28,-18,-8,24,-26,11,15,-11,15,9,-18,6,6,19,27,-7,20,-44,8,4,-18,4,24,-17,6,1,-1,-33,-12,-36,-33,2,-8,13,25,-6,10,8,18,7,6,13,-22,-6,-25,-15,-26,20,-25,24,28,-11,12,0,-18,5,-20,18,-25,3,-33,1,-21,20,-32,34,9,-35,-7,-13,-1,2,-37,-3,-24,2,-32,7,-7,13,1,3,20,-72,-8,-9,-18,2,-36,6,-2,1,-17,7,-12,25,25,8,-14,-14,11,-12,-1,10,-6,2,9,-32,-22,-24,-33,22,-17,19,6,2,-4,0,-10,1,3,13,-5,-3,-48,-8,-11,0,-39,7,2,5,26,10,-22,-2,2,6,-14,8,-40,10,19,-19,-14,15,-1,-11,14,1,-33,28,-8,-3,-22,15,-68,20,18,-13,16,-9,0,-1,-4,-18,1,-8,-5,-14,1,5,-11,37,32,-1,27,-22,-50,27,11,-35,19,-16,12,12,-9,-3,-7,14,-68,14,-13,4,5,-6,11,13,-1,13,18,-37,12,-11,9,5,-8,-1,-48,9,6,5,11,-6,-3,-19,0,-42,21,2,2,10,12,1,-6,-2,26,10,15,-3,-21,17,4,-21,-3,-2,-15,24,15,1,-4,5,-16,14,4,-9,-1,-1,19,1,17,11,-9,31,-17,17,12,-16,-12,11,16,12,-13,-14,15,4,9,18,24,16,-45,6,-6,13,9,-7,-5,1,7,5,12,-23,5,7,9,2,-4,-20,-26,2,11,3,1,-7,0,8,-4,-2,-10,6,8,6,11,-33,-2,-23,15,3,-11,3,-7,-6,10,-18,-15,-11,3,-2,2,-14,-2,19,1,0,-26,7,-14,-12,19,-14,5,6,-2,-26,-11,5,-4,-7,10,9,-8,6,-9,2,-7,-13,7,25,-1,3,-13,23,-2,-21,3,3,-26,-10,11,-5,7,-15,-52,16,-14,9,-2,-33,16,-26,8,-15,-21,-7,10,8,10,-27,-26,6,-12,-1,26,-32,22,-5,6,-6,-40,-6,8,23,10,-29,25,-23,3,-26,4,6,15,17,3,-14,-3,2,-5,-3,-9,-6,16,-17,-2,-31,1,-17,-3,-17,-6,-13,12,8,-1,-8,-12,-14,11,0,-9,-11,-3,-1,2,3,-14,-22,-9,-5,1,-26,5,0,17,-4,16,2,-22,-8,-12,-2,8,0,-3,24,16,-2,0,15,12,-27,-5,14,-6,-24,-3,9,12,-8,4,12,19,12,-10,22,-7,-19,-12,21,9,-11,-6,9,10,10,-10,-16,-10,8,-15,-12,-19,-2,-26,1,-9,-2,-9,3,21,-1,5,16,4,-3,9,9,-48,-3,-9,27,13,-4,-6,21,-1,-12,-1,7,-12,-11,-20,17,-14,-7,-16,7,6,-19,-19,-38,18,-16,-5,-12,17,-37,-40,5,6,-22,-1,16,3,11,24,3,-30,-10,-11,-8,-3,-11,-13,1,18,-14,0,7,-5,5,7,11,-9,21,14,12,-29,13,-6,7,25,11,11,23,-7,11,6,-9,-23,-5,-34,-9,-15,1,-25,-4,-8,-9,2,4,17,13,14,13,-11,7,11,-26,-23,1,-8,-16,5,-2,-38,-13,26,-3,-8,-11,8,-2,-9,-4,-14,13,-16,-23,11,13,-14,5,5,-1,5,15}
#define FC1_WT_SHAPE 1000
#define FC1_BIAS {46,7,27,23,3,-67,-3,-7,-25,-49}
#define FC1_BIAS_SHAPE 10
