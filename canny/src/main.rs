
use std::{ f32::consts::PI, ops::Div, cmp};

use image::{GenericImageView, DynamicImage, GenericImage,ImageBuffer,Rgb, Rgb32FImage, GrayImage};
use rayon::{iter::{IntoParallelIterator, ParallelIterator, ParallelBridge, IntoParallelRefIterator, IndexedParallelIterator}, slice::{ParallelSlice, ParallelSliceMut}};
//use ndarray_image::{self, open_image};

fn get_str_ascii(intent :u8)-> &'static str{
    let index = intent/32;
    let ascii = [" ",".",",","-","~","+","=","@"];
    return ascii[index as usize];
}

const SOBEL_V:[[i16;3];3] = [[-1 , -2, -1], [0,0,0],[1,2,1]];
const SOBEL_H:[[i16;3];3] = [[-1,0,1],[-2,0,2],[-1,0,1]];
const ONE:i8 = -1;

fn convol(img:&GrayImage, kernel:&[[i16;3]],x: u32,y:u32) -> i16{
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
    //a.abs() as u16
    a
}

/*const Quanz:f32 = PI.div(8f32);
const Quanz2:f32 = PI.div(4f32);*/
fn quantized(zeta: f32) -> u8 {
    let thres = (zeta.div(PI)*180.0).round().div(45.0);
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

fn get_image(dir: &str) -> (GrayImage, GrayImage){
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
            let pyth: u8 = (((value.abs() as u16 + value2.abs() as u16)*25)/100) as u8;
            let angle = fast_math::atan2(value as f32, value2 as f32);
            let angle = quantized(angle);
            //print!("({:})",angle);
            *ang_img.get_pixel_mut(x-1, y-1) = image::Luma([angle*20]);
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
    n_img.save("output.png").unwrap();
    ang_img.save("phase.png").unwrap();
    (ang_img, n_img)
}

fn non_max_sup(sobeled: &mut GrayImage, phase: &mut GrayImage){
    fn pix(x:u32,y:u32, sobeled: &GrayImage) -> u8 {
        sobeled.get_pixel(x, y).0[0]
    }
    let (width,height) = sobeled.dimensions();
    for x in 1..width-1{
        for y in 1..height-1{
            let angle = phase.get_pixel(x, y).0[0]/20;
            let h = pix(x-1, y, sobeled) <= pix(x, y, sobeled) && pix(x, y, sobeled) >= pix(x+1, y, sobeled);
            let dia_1 = pix(x-1, y-1, sobeled) <= pix(x, y, sobeled) && pix(x, y, sobeled) >= pix(x+1, y+1, sobeled);
            let dia_3 = pix(x-1, y+1, sobeled) <= pix(x, y, sobeled) && pix(x, y, sobeled) >= pix(x+1, y-1, sobeled);
            let v = pix(x, y+1, sobeled) <= pix(x, y, sobeled) && pix(x, y, sobeled) >= pix(x, y-1, sobeled);
            
            match angle{
                0 =>   if h == false{
                    *sobeled.get_pixel_mut(x, y) = image::Luma([0]);
                },
                1 =>if dia_1 == false{
                    *sobeled.get_pixel_mut(x, y) = image::Luma([0]);
                },
                2=>if dia_3 == false{
                    *sobeled.get_pixel_mut(x, y) = image::Luma([0]);
                },
                3=>if v == false{
                    *sobeled.get_pixel_mut(x, y) = image::Luma([0]);
                },
                _=> panic!(),
            }


        }
    }
    sobeled.save("non_max2.png").unwrap();
    //let so = sobeled;
    let non_max = normalize(sobeled);
    //println!("{:?}",non_max);
    //println!("NEW");
    thres(sobeled);

}

fn normalize(sobel: & GrayImage) -> Vec<f32>{
    //let m = vec![1,3,3];
    let a:Vec<u8> = sobel.par_iter().map(|p| *p).collect();
    //println!("{:?}",a);
    //println!("vectorize");
    //let max = sobel.pixels().par_bridge().map(|p| p.0[0]).max().unwrap();
    //let min = sobel.pixels().par_bridge().map(|p| p.0[0]).min().unwrap();
    let min = sobel.par_iter().map(|p| *p).min().unwrap();
    let max = sobel.par_iter().map(|p| *p).max().unwrap();
    //println!("min ={}", min);
    let change: Vec<f32>= sobel.par_chunks(10).flat_map(|p|{
            //let d:Vec<u8>  = (*p).to_vec();
            let fin:Vec<f32> = p.into_iter().map(|p2|{
                let v = (*p2 - min) as f32;
                let v = (v).div((max - min) as f32);
                v
            }).collect();
            fin
        }).collect();
    /*let change:Vec<f32> = sobel.pixels().par_bridge().map(|p|{
        let v =(p.0[0] - min) as f32;
        let v = (v).div((max - min) as f32);
        v
    }).collect();*/
    change



}
fn thres(non_max: &mut GrayImage){
    let (width,height) = non_max.dimensions();
    let width = width as usize;
    let non = normalize(non_max);
    let _ = non_max.enumerate_pixels_mut().into_iter().for_each(|(x,y,p)|{
        //print!("({})",non[(y*x + x) as usize]);
        //non[(y*x + x) as usize]
        //*p = image::Luma([(non[(y*x + x) as usize] as u8 * 40)]);
        if non[(x*y + y) as usize] > 0.7{
            *p = image::Luma([(255)]);
        }else if non[(y*x + x) as usize] < 0.3{
            *p = image::Luma([(0)]);
        }else{
            *p = image::Luma([(125)]);
        }
    });

    let e = vec![12,3,3];
    // what is we use zip instead??
    let v = non.par_chunks(width).zip(non_max.par_chunks_mut(width)).for_each(
        |(vec,img)|
            {
            vec.iter().zip((*img).iter_mut()).for_each(
                |(v1,m2)|{
                    let v1 = *v1;
                    if v1 > 0.7{
                        *m2 = 255
                    }else if v1 < 0.3{
                        *m2 = 0
                    }else{
                        *m2 = 50
                    }
                }
            )

        }
    );
    /* 
    non.par_chunks_mut(10).for_each(|p|{
        //let d:Vec<u8>  = (*p).to_vec();
        let fin:Vec<f32> = p.into_iter().for_each(|p2|{
            let p3 = *p2;
            if p3 > 0.7 {
                *p2 = 255;
            }else if p3 < 0.3{
                *p2 = 0;
            }else{
                *p3 = 122;
            }
        }).collect();
        fin
    }).collect(); */
    non_max.save("thres.png").unwrap();
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
    
    let (mut phase,mut sobel) = get_image("../brick-house.png");
    println!("{}",SOBEL_H[1][2]);
    non_max_sup(& mut sobel, & mut phase);
    //let n = image::open("non_max2.png").unwrap().to_luma8();
    //let n = normalize(&n);
    //println!("{:?}",n)



    //non_max_sup(sobeled, phase)


}
