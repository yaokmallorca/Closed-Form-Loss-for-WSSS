# Closed-Form-Loss-for-WSSS
A DCNN-based closed-form loss for scribble-based weakly supervised semantic segmentation. 
This work is inspired by "Closed-Form Matting"
A. Levin, D. Lischinski and Y. Weiss, "A Closed-Form Solution to Natural Image Matting," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 30, no. 2, pp. 228-242, Feb. 2008, doi: 10.1109/TPAMI.2007.1177.

![alt text](https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/closed-form-segmentation.png?raw=true)
CityScapes example

<table>
  <tr>
    <td> <img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_erfurt_000040_000019_leftImg8bit.png" width=200 height=120 ></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_erfurt_000040_000019_leftImg8bit_alpha_car.png" width=200 height=120></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_erfurt_000040_000019_leftImg8bit_alpha_construction.png" width=200 height=120></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_erfurt_000040_000019_leftImg8bit_alpha_person.png" width=200 height=120></td>
   </tr>
  <tr>
    <td> <img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_erfurt_000040_000019_leftImg8bit_alpha_plant.png" width=200 height=120 ></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_erfurt_000040_000019_leftImg8bit_alpha_road.png" width=200 height=120></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_erfurt_000040_000019_leftImg8bit_alpha_sidewalk.png" width=200 height=120></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_erfurt_000040_000019_leftImg8bit_alpha_sky.png" width=200 height=120></td>
    </tr>
  
  <tr>
    <td> <img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_hamburg_000000_048750_leftImg8bit.png" width=200 height=120 ></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_hamburg_000000_048750_leftImg8bit_alpha_car.png" width=200 height=120></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_hamburg_000000_048750_leftImg8bit_alpha_construction.png" width=200 height=120></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_hamburg_000000_048750_leftImg8bit_alpha_person.png" width=200 height=120></td>
   </tr>
  <tr>
    <td> <img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_hamburg_000000_048750_leftImg8bit_alpha_plant.png" width=200 height=120 ></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_hamburg_000000_048750_leftImg8bit_alpha_road.png" width=200 height=120></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_hamburg_000000_048750_leftImg8bit_alpha_sidewalk.png" width=200 height=120></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_cityscape_hamburg_000000_048750_leftImg8bit_alpha_sign.png" width=200 height=120></td>
    </tr>
</table>

VOC example
<table>
  <tr>
    <td> <img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_voc_2007_000129.png" width=200 height=120 ></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_voc_2007_000129_scr.png" width=200 height=120></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_voc_2007_000129_alpha_bike.png" width=200 height=120></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_voc_2007_000129_alpha_person.png" width=200 height=120></td>
   </tr>
  <tr>
    <td> <img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_voc_2007_001763.png" width=200 height=120 ></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_voc_2007_001763_scr.png" width=200 height=120></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_voc_2007_001763_alpha_cat.png" width=200 height=120></td>
    <td><img src="https://github.com/yaokmallorca/Closed-Form-Loss-for-WSSS/blob/master/imgs/results_voc_2007_001763_alpha_dog.png" width=200 height=120></td>
   </tr>
 
  </table>
 Todo:
   1. Deep Neural Network to solve "Learning based digital matting" 
   Y. Zheng and C. Kambhamettu, "Learning based digital matting," 2009 IEEE 12th International Conference on Computer Vision, 2009, pp. 889-896, doi: 10.1109/ICCV.2009.5459326.
