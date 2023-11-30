
use std::{ f32::consts::PI, ops::Div};

use image::{GenericImageView, DynamicImage, GenericImage,ImageBuffer,Rgb, Rgb32FImage, GrayImage};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
//use ndarray_image::{self, open_image};

fn get_str_ascii(intent :u8)-> &'static str{
    let index = intent/32;
    let ascii = [" ",".",",","-","~","+","=","@"];
    return ascii[index as usize];
}

const SOBEL_V:[[i16;3];3] = [[-1 , -2, -1], [0,0,0],[1,2,1]];
const SOBEL_H:[[i16;3];3] = [[-1,0,1],[-2,0,2],[-1,0,1]];
const ONE:i8 = -1;

fn convol(img:&GrayImage, kernel:&[[i16;3]],x: u32,y:u32) -> u16{
    let y = y -1;
    let x1 = 1;
    let y1: usize = 0;
    let a1_1: i16 = img.get_pixel(x-1, y).0[0] as i16 * kernel[x1-1][y1];
    let a2_1 = img.get_pixel(x, y).0[0] as i16 * kernel[x1][y1];
    let a3_1 = img.get_pixel(x+1, y).0[0] as i16 * kernel[x1+1][y1];
    let (y,y1) = (y +1, y1+1);
    let a1_2 = img.get_pixel(x-1, y).0[0] as i16 * kernel[x1-1][y1];
    let a2_2 = img.get_pixel(x, y).0[0] as i16 * kernel[x1][y1];
    let a3_2 = img.get_pixel(x+1, y).0[0] as i16 * kernel[x1+1][y1];
    let (y,y1) = (y +1, y1+1);
    let a1_3 = img.get_pixel(x-1, y).0[0] as i16 * kernel[x1-1][y1];
    let a2_3 = img.get_pixel(x, y).0[0] as i16 * kernel[x1][y1];
    let a3_3 = img.get_pixel(x+1, y).0[0] as i16 * kernel[x1+1][y1];
    /*let a1_1: i16 = img.get_pixel(x-1, y).0[0] as i16 ;
    let a2_1 = img.get_pixel(x, y).0[0] as i16 ;
    let a3_1 = img.get_pixel(x+1, y).0[0] as i16 ;
    let (y,y1) = (y +1, y1+1);
    let a1_2 = img.get_pixel(x-1, y).0[0] as i16 ;
    let a2_2 = img.get_pixel(x, y).0[0] as i16 ;
    let a3_2 = img.get_pixel(x+1, y).0[0] as i16 ;
    let (y,y1) = (y +1, y1+1);
    let a1_3 = img.get_pixel(x-1, y).0[0] as i16 ;
    let a2_3 = img.get_pixel(x, y).0[0] as i16 ;
    let a3_3 = img.get_pixel(x+1, y).0[0] as i16 ;*/
    let a:Vec<i16> = vec![a1_1,a2_1,a3_1,a1_2,a2_2,a3_2,a1_3,a2_3,a3_3];
    //println!("({:?})",a);
    let a:i16 =a.into_iter().sum();
    //println!("sum = ({:?})",a);
    a.abs() as u16
}

/*const Quanz:f32 = PI.div(8f32);
const Quanz2:f32 = PI.div(4f32);*/
fn quantized(zeta: f32) -> u8 {
    let thres = zeta.div(PI).round().div(45.0);
    if thres >= 0.0 {
        let degree: u8 = if zeta.div(PI).round().abs() % 45f32 < 22.5{
            0
        }else{
            1
        };
        let ans:Option<u8> = match thres {
            x if x < 1.0 => Some(0 + degree), // < 45
            x if x < 2.0 => Some(1 + degree), // < 90
            x if x < 3.0 => Some(2 + degree), // < 135
            x if x <= 4.0 => if degree == 0{
                Some(3)
            }else{
                Some(0)
            },
            _ => None,
        };
        ans.unwrap()

    }else{
        let degree: u8 = if zeta.div(PI).round().abs() % 45f32 < 22.5{
            1
        }else{
            0
        };
        let ans:Option<u8> = match thres {
            x if x > -1.0 => if degree == 0{
                Some(3)
            }else{
                Some(0)
            }, // < 45
            x if x > -2.0 => Some(2 + degree), // < 90
            x if x > -3.0 => Some(1 + degree), // < 135
            x if x >= -4.0 => Some(0 + degree),
            _ => None,
        };
        ans.unwrap()
    }
    /*if  -Quanz <= zeta && zeta <= Quanz{

    }else if Quan <= zeta <= (Quanz2 + Quanz){

    }else if (Quanz2 + Quanz) <= (2*Quanz + Quanz){


    }else if PI+ Quanz >= zeta >= PI - Quanz2 - Quanz{

    }else*/
}

fn get_image(dir: &str) -> GrayImage{
    let img = image::open(dir).unwrap().to_luma8();
    println!("{:?}", img.dimensions());
    let (width,height) = img.dimensions();
    let mut n_img: GrayImage = ImageBuffer::new(width-2,height-2);
    let mut ang_img: GrayImage = ImageBuffer::new(width-2,height-2);
    for x in 1..width-1{
        for y in 1..height-1{
            //print!("")
            let val = img.get_pixel(x, y);
            let value = convol(&img, &SOBEL_H, x, y);
            let value2 = convol(&img, &SOBEL_V, x, y);
            //let ans: u8 = value.pow(2) + value2.pow(2);
            let pyth: u8 = (((value + value2)*25)/100) as u8;
            let angle = fast_math::atan2(value as f32, value2 as f32);
            let angle = quantized(angle);
            *ang_img.get_pixel_mut(x-1, y-1) = image::Luma([angle]);

            *n_img.get_pixel_mut(x-1, y-1) = image::Luma([pyth]);
            //print!("({:?})",val);

        }
        //println!("\nnewline");

    }
    //let p2 = p.get_pixel(0, 0);
    /*for y in 0..height{
        for x in 0..width{
            let pix: &image::Luma<f32> = img.get_pixel(x, y);
            let (b,g,r) =  (pix[0] ,pix[1],pix[2]);
            let n = (r_coef * r, g_coef * g, b_coef * b);

            *image.get_pixel_mut(x, y) = image::Luma([ (n.0 * 255.0), (n.1 * 255.0) , (n.2 * 255.0)]);
            
            /*if y % (scale * 2) == 0 && x % scale ==0{
                let pix = img.get_pixel(x,y);
                let mut intent = pix[0]/3 + pix[1]/3 + pix[2]/3;
                if pix[3] ==0{
                    intent = 0;
                }
                print!("{}",get_str_ascii(intent));
            }*/
        }
        /*if y%(scale*2)==0{
            println!("");
        }*/
    }*/
    n_img.save("output3.png").unwrap();
    ang_img
}



fn main() {
    println!("Hello, world!");
    //let mut img: DynamicImage = image::open("../brick-house.png").unwrap();
    //let img2 = open_image("../brick-house.png", ndarray_image::Colors::Bgra).expect("unable to open input image");
    //let mut image = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(10, 10);
    
    /*let mut image = GrayImage::new(10, 10);
    for (x,y,pixel) in image.enumerate_pixels_mut(){
        if x == 1{
            *pixel = image::Luma([200]);
        }else{
            *pixel = image::Luma([1]);
        }
    }
    //image.get_pixel_mut(5, 5) = image::Rgb([255, 255, 255]);
    image.save("sample.png").unwrap();
    //println!("{:?}",img);
    //get_image("../sample.jpg");*/
    
    let angle = get_image("sample.png");
    angle.save("phase").unwrap();
    println!("{}",SOBEL_H[1][2]);

}
