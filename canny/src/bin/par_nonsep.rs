use std::{ f32::consts::PI, ops::Div, cmp, cell::UnsafeCell, sync::{Arc, Mutex, RwLock, mpsc}};

use image::{GenericImageView, DynamicImage, GenericImage,ImageBuffer,Rgb, Rgb32FImage, GrayImage};
use rayon::{iter::{IntoParallelIterator, ParallelIterator, ParallelBridge, IntoParallelRefIterator, IndexedParallelIterator, IntoParallelRefMutIterator}, slice::{ParallelSlice, ParallelSliceMut}};
//use ndarray_image::{self, open_image};
use image::Luma;


const SOBEL_V:[[i16;3];3] = [[-1 , -2, -1], [0,0,0],[1,2,1]];
const SOBEL_H:[[i16;3];3] = [[-1,0,1],[-2,0,2],[-1,0,1]];
const ONE:i8 = -1;
const GUSS:[i16;5] = [1,4,6,4,1];
const GUSS2:[[i32;5];5] = [[1,4,6,4,1],[4,16,24,16,4],[6,24,32,24,6],[4,16,24,16,4],[1,4,6,4,1]];

fn gussian(img:&mut GrayImage) -> GrayImage{
    let (width,height) = img.dimensions();
    let size =  ((width-2)*(height-2)) as usize;
    let mut n_img: GrayImage = ImageBuffer::new(width-4,height-4);
    //let mut gus_v: Vec<f32> = Vec::with_capacity(size);
    for x in 2..width-2{
        for y in 2..height-2{
            
            let value = (convol1D_5(img, &GUSS, x, y, 0) as f32).div(16f32);
            *img.get_pixel_mut(x, y) = image::Luma([value as u8]);
            
            
        }
    }
    for x in 2..width-2{
        for y in 2..height-2{
            
            let value = (convol1D_5(img, &GUSS, x, y, 1) as f32).div(16f32);
            *img.get_pixel_mut(x, y) = image::Luma([value as u8]);
            
        }
    }
    for x in 2..width-2{
        for y in 2..height-2{
            *n_img.get_pixel_mut(x-2, y-2) = *img.get_pixel(x, y);
        }

    }
    
    n_img.save("par_guss.png").unwrap();
    n_img

}

fn par_convol2D_5(img:&mut GrayImage, kernel:&[[i32;5];5]) -> GrayImage{
    let (width,height) = img.dimensions();
    let mut n_img: GrayImage = ImageBuffer::new(width,height);
    n_img.enumerate_pixels_mut().par_bridge().for_each(|p|
        {
            let mut sum: i32 = 0;
            if (p.0 >=2 && p.0 < width-2) && ( p.1 >=2 && p.1 < height-2)
            {
                for x1 in 0..5{
                    for y1 in 0..5{
                        //let v = p.2[0];
                        sum += img.get_pixel(p.0 -2 + x1, p.1 -2 + y1).0[0] as i32 * kernel[x1 as usize][y1 as usize];
                    }
                    *p.2 = image::Luma([(sum as f32).div(256f32) as u8]);
                }
            }
        });
    let n_img = n_img.sub_image(2, 2, width-4, height-4).to_image();
    n_img.save("par_guss.png").unwrap();
    n_img

}

fn convol1D_5(img:&GrayImage, kernel:&[i16;5],x:u32,y:u32,xy:u8) -> i16{
    if xy == 0{
        let a1_1 = img.get_pixel(x, y).0[0] as i16 * kernel[2];
        let a1_2 = img.get_pixel(x-1, y).0[0] as i16 * kernel[1];
        let a1_3 = img.get_pixel(x+1, y).0[0] as i16 * kernel[3];
        let a1_4 = img.get_pixel(x-2, y).0[0] as i16 * kernel[0];
        let a1_5 = img.get_pixel(x+2, y).0[0] as i16 * kernel[4];
        let s = a1_1 + a1_2 + a1_3 + a1_4 + a1_5;
        s

    }else{
        let a1_1 = img.get_pixel(x, y).0[0] as i16 * kernel[2];
        let a1_2 = img.get_pixel(x, y-1).0[0] as i16 * kernel[1];
        let a1_3 = img.get_pixel(x, y+1).0[0] as i16 * kernel[3];
        let a1_4 = img.get_pixel(x, y-2).0[0] as i16 * kernel[0];
        let a1_5 = img.get_pixel(x, y+2).0[0] as i16 * kernel[4];
        let s = a1_1 + a1_2 + a1_3 + a1_4 + a1_5;
        s
    }
}

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
    
    let a:Vec<i16> = vec![a1_1,a2_1,a3_1,a1_2,a2_2,a3_2,a1_3,a2_3,a3_3];
    
    let a:i16 =a.into_iter().sum();
    
    a
}


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

#[derive(Copy, Clone)]
pub struct UnsafeSlice<'a, T> {
    slice: &'a [UnsafeCell<T>],
}
unsafe impl<'a, T: Send + Sync> Send for UnsafeSlice<'a, T> {}
unsafe impl<'a, T: Send + Sync> Sync for UnsafeSlice<'a, T> {}

impl<'a, T> UnsafeSlice<'a, T> {
    pub fn new(slice: &'a mut [T]) -> Self {
        let ptr = slice as *mut [T] as *const [UnsafeCell<T>];
        Self {
            slice: unsafe { &*ptr },
        }
    }
    
    /// SAFETY: It is UB if two threads write to the same index without
    /// synchronization.
    pub unsafe fn write(&self, i: usize, value: T) {
        let ptr = self.slice[i].get();
        *ptr = value;
    }
}

fn get_image(dir: &str) -> (GrayImage, GrayImage){
    let mut img = image::open(dir).unwrap().to_luma8();
    let img = par_convol2D_5(&mut img, &GUSS2);
    //let img = gussian(&mut img);
    println!("{:?}", img.dimensions());
    let (width,height) = img.dimensions();
    //gussian(&mut img);
    let mut n_img = ImageBuffer::new(width,height);
    let mut ang_img = ImageBuffer::new(width,height);
    
            let mut m: Vec<u8> = Vec::with_capacity(n_img.len());
            unsafe{ m.set_len(n_img.len())}
            let l = n_img.clone().len();
            let m = n_img.as_parallel_slice_mut();
            let n = ang_img.as_parallel_slice_mut();
            println!("len {}",l);
            let point = UnsafeSlice::new(m);
            let point2 = UnsafeSlice::new(n);
            //(436, 596)
            (1..(width-1)).into_iter().for_each(|x|
                {
                    (1..(height-1)).into_par_iter().for_each(|y|
                        {
                        
                        let value = convol(&img, &SOBEL_H, x, y);
                        let value2 = convol(&img, &SOBEL_V, x, y);
                        let pyth: u8 = (((value.abs() as u16 + value2.abs() as u16)*25)/100) as u8;
                        let angle = fast_math::atan2(value as f32, value2 as f32);
                        let angle = quantized(angle);
                        
                        unsafe{
                            point.write((x + (width)*y) as usize, pyth);
                            point2.write((x + (width)*y) as usize, angle*20);
                        }
                        });

                    
                });
            //println!("{:?}",m);
    //     n_img.enumerate_pixels_mut().zip(ang_img.enumerate_pixels_mut()).par_bridge().for_each(|(p1,p2)|{
    //     let x = p1.0;
    //     let y = p1.1;
    //     let (value,value2) = rayon::join(|| convol(&img, &SOBEL_H, x+2, y+2), || convol(&img, &SOBEL_V, x+2, y+2));
    //     let pyth: u8 = (((value.abs() as u16 + value2.abs() as u16)*25)/100) as u8;
    //     let angle = fast_math::atan2(value as f32, value2 as f32);
    //     let angle = quantized(angle);
    //     unsafe{
    //         *p1.2 = image::Luma([pyth]);
    //         *p2.2 = image::Luma([angle*20]);
    //     }


    // });
    
        // let mut nim = Arc::new(Mutex::new(n_img));
        // let mut anim = Arc::new(Mutex::new(ang_img));
        

    // let r: &mut [u8] = n_img.as_parallel_slice_mut();
    // let r2 = ang_img.as_parallel_slice_mut();
    // let a = UnsafeSlice::new(r);
    // let a2 = UnsafeSlice::new(r2);
    //let r: rayon::slice::Iter<'_, u8> = ang_img.into_par_iter();

        // (1..(width-1)).into_iter().for_each(|x|
        //         {
        //             (1..(height-1)).into_par_iter().for_each(|y|
        //                 {
        //                     let (value,value2) = rayon::join(|| convol(&img, &SOBEL_H, x, y), || convol(&img, &SOBEL_V, x, y));
        //                     let pyth: u8 = (((value.abs() as u16 + value2.abs() as u16)*25)/100) as u8;
        //                     let angle = fast_math::atan2(value as f32, value2 as f32);
        //                     let angle = quantized(angle);
        //                     unsafe {
        //                         // a.write(((x-1) + y*(width-3)) as usize, angle*20);
        //                         // a2.write(((x-1) + y*(width-3)) as usize, pyth);
                                
        //                         *nim.lock().unwrap().get_pixel_mut(x-1, y-1) = image::Luma([pyth]);
        //                         *anim.lock().unwrap().get_pixel_mut(x-1, y-1) = image::Luma([angle*20]);
        //                     }

        //                 }
        //             )}
        //         );

    // let r = n_img.into_par_iter().zip(r).chunks(width as usize).enumerate()
    // .for_each(|(x,p)|{
    //     let x:u32 = x as u32;
    //     p.into_iter_().enumerate().for_each(|(y,p)|{
    //         let (pix_n,pix_a) = p;
    //         let (value,value2) = rayon::join(|| convol(&img, &SOBEL_H, x, y as u32), || convol(&img, &SOBEL_V, x, y as u32));
    //         let pyth: u8 = (((value.abs() as u16 + value2.abs() as u16)*25)/100) as u8;
    //         let angle = fast_math::atan2(value as f32, value2 as f32);
    //         pu

    //     })
    // });
    // let mut n_img_p= (n_img.par_chunks_mut(width as usize)).zip(ang_img.par_chunks_mut(width as usize)).enumerate()
    // .into_par_iter().for_each(|(x,(p1,p2))|{
    //     let x: u32 = x as u32;
    //     p1.into_iter().zip(p2.into_iter()).into_iter().enumerate().for_each(|(y,(pix1,pix2))|{
    //         let (value,value2) = rayon::join(|| convol(&img, &SOBEL_H, x, y as u32), || convol(&img, &SOBEL_V, x, y as u32));
    //         let pyth: u8 = (((value.abs() as u16 + value2.abs() as u16)*25)/100) as u8;
    //         let angle = fast_math::atan2(value as f32, value2 as f32);
    //         let angle = quantized(angle);
    //         *pix2 = angle*20;
    //         *pix1 = pyth;
    //     })
    // });

    // (1..(width-1)).into_iter().for_each(|x|
    //     {
    //         (1..(height-1)).into_iter().for_each(|y|
    //             {
    //                 // let value = convol(&img, &SOBEL_H, x, y);
    //                 // let value2 = convol(&img, &SOBEL_V, x, y);
    //                 let (value,value2) = rayon::join(|| convol(&img, &SOBEL_H, x, y), || convol(&img, &SOBEL_V, x, y));
    //                 let pyth: u8 = (((value.abs() as u16 + value2.abs() as u16)*25)/100) as u8;
    //                 let angle = fast_math::atan2(value as f32, value2 as f32);
    //                 let angle = quantized(angle);
    //                 *ang_img.get_pixel_mut(x-1, y-1) = image::Luma([angle*20]);
    //                 unsafe {
    //                     ang_img.unsafe_put_pixel(x-1, y-1, image::Luma([angle*20]));
    //                 }
    //                 *n_img.get_pixel_mut(x-1, y-1) = image::Luma([pyth]);
    //             });
    //     });
    // for x in 1..width-1{
    //     for y in 1..height-1{
    //         //print!("")
    //         let val = img.get_pixel(x, y);
    //         let value = convol(&img, &SOBEL_H, x, y);
    //         let value2 = convol(&img, &SOBEL_V, x, y);
    //         let pyth: u8 = (((value.abs() as u16 + value2.abs() as u16)*25)/100) as u8;
    //         let angle = fast_math::atan2(value as f32, value2 as f32);
    //         let angle = quantized(angle);
    //         *ang_img.get_pixel_mut(x-1, y-1) = image::Luma([angle*20]);
    //         *n_img.get_pixel_mut(x-1, y-1) = image::Luma([pyth]);

    //     }

    // }

    // let n_img = Arc::try_unwrap(nim).unwrap().into_inner().unwrap();
    // let ang_img = Arc::try_unwrap(anim).unwrap().into_inner().unwrap();
    
    n_img.save("par_output.png").unwrap();
    ang_img.save("par_phase.png").unwrap();
    (ang_img, n_img)
    
}

fn non_max_sup(sobeled: &mut GrayImage, phase: &mut GrayImage){
    let (width,height) = sobeled.dimensions();
    //let mut sobeled2: GrayImage = ImageBuffer::new(width, height);
    //let sobeled = &mut sobeled;
    //let mut binding = sobeled;
    let mut sobeled2 = &sobeled.clone();
    let m = sobeled.as_parallel_slice_mut();
    let point = UnsafeSlice::new(m);
    fn pix(x:u32,y:u32, sobeled: &GrayImage) -> u8 {
        sobeled.get_pixel(x, y).0[0]
    }
    //let (width,height) = sobeled.dimensions();
    (1..(width-1)).into_iter().for_each(|x|
        {
            (1..(height-1)).into_par_iter().for_each(|y|
                {
                    let angle = phase.get_pixel(x, y).0[0]/20;
                    let h = pix(x-1, y, sobeled2) <= pix(x, y, sobeled2) && pix(x, y, sobeled2) >= pix(x+1, y, sobeled2);
                    let dia_1 = pix(x-1, y-1, sobeled2) <= pix(x, y, sobeled2) && pix(x, y, sobeled2) >= pix(x+1, y+1, sobeled2);
                    let dia_3 = pix(x-1, y+1, sobeled2) <= pix(x, y, sobeled2) && pix(x, y, sobeled2) >= pix(x+1, y-1, sobeled2);
                    let v = pix(x, y+1, sobeled2) <= pix(x, y, sobeled2) && pix(x, y, sobeled2) >= pix(x, y-1, sobeled2);
                    
                    match angle{
                        0 =>   if h == false{
                            //*sobeled.get_pixel_mut(x, y) = image::Luma([0]);
                            unsafe{
                                point.write((x + (width)*y) as usize, 0);
                            }
                        }/*else{
                            *sobeled.get_pixel_mut(x, y) = *sobeled.get_pixel(x, y);
                        }*/,
                        1 =>if dia_1 == false{
                            //*sobeled.get_pixel_mut(x, y) = image::Luma([0]);
                            unsafe{
                                point.write((x + (width)*y) as usize, 0);
                            }
                        }/*else{
                            *sobeled.get_pixel_mut(x, y) = *sobeled.get_pixel(x, y);
                        }*/,
                        2=>if dia_3 == false{
                            //*sobeled.get_pixel_mut(x, y) = image::Luma([0]);
                            unsafe{
                                point.write((x + (width)*y) as usize, 0);
                            }
                        }/*else{
                            *sobeled.get_pixel_mut(x, y) = *sobeled.get_pixel(x, y);
                        }*/,
                        3=>if v == false{
                            //*sobeled.get_pixel_mut(x, y) = image::Luma([0]);
                            unsafe{
                                point.write((x + (width)*y) as usize, 0);
                            }
                        }/*else{
                            *sobeled.get_pixel_mut(x, y) = *sobeled.get_pixel(x, y);
                        }*/,
                        _=> panic!(),
                    }
                        });
        });
    // for x in 1..width-1{
    //     for y in 1..height-1{
    //         let angle = phase.get_pixel(x, y).0[0]/20;
    //         let h = pix(x-1, y, sobeled) <= pix(x, y, sobeled) && pix(x, y, sobeled) >= pix(x+1, y, sobeled);
    //         let dia_1 = pix(x-1, y-1, sobeled) <= pix(x, y, sobeled) && pix(x, y, sobeled) >= pix(x+1, y+1, sobeled);
    //         let dia_3 = pix(x-1, y+1, sobeled) <= pix(x, y, sobeled) && pix(x, y, sobeled) >= pix(x+1, y-1, sobeled);
    //         let v = pix(x, y+1, sobeled) <= pix(x, y, sobeled) && pix(x, y, sobeled) >= pix(x, y-1, sobeled);
            
    //         match angle{
    //             0 =>   if h == false{
    //                 *sobeled.get_pixel_mut(x, y) = image::Luma([0]);
    //             }/*else{
    //                 *sobeled.get_pixel_mut(x, y) = *sobeled.get_pixel(x, y);
    //             }*/,
    //             1 =>if dia_1 == false{
    //                 *sobeled.get_pixel_mut(x, y) = image::Luma([0]);
    //             }/*else{
    //                 *sobeled.get_pixel_mut(x, y) = *sobeled.get_pixel(x, y);
    //             }*/,
    //             2=>if dia_3 == false{
    //                 *sobeled.get_pixel_mut(x, y) = image::Luma([0]);
    //             }/*else{
    //                 *sobeled.get_pixel_mut(x, y) = *sobeled.get_pixel(x, y);
    //             }*/,
    //             3=>if v == false{
    //                 *sobeled.get_pixel_mut(x, y) = image::Luma([0]);
    //             }/*else{
    //                 *sobeled.get_pixel_mut(x, y) = *sobeled.get_pixel(x, y);
    //             }*/,
    //             _=> panic!(),
    //         }


    //     }
    // }
    sobeled.save("par_non_max2.png").unwrap();
    //hyst(sobeled);
    //let so = sobeled;
    let non_max = normalize(sobeled);
    //println!("{:?}",non_max);
    //println!("NEW");
    thres(sobeled);
    hyst(sobeled);

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
    let (width,height) = sobel.dimensions();
    let change: Vec<f32>= sobel.par_chunks(width as usize).flat_map(|p|{
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

    /* */
    // let _ = non_max.enumerate_pixels_mut().into_iter().for_each(|(x,y,p)|{
    //     //print!("({})",non[(y*x + x) as usize]);
    //     //non[(y*x + x) as usize]
    //     //*p = image::Luma([(non[(y*x + x) as usize] as u8 * 40)]);
    //     if non[(x*y + y) as usize] > 0.7{
    //         *p = image::Luma([(255)]);
    //     }else if non[(y*x + x) as usize] < 0.3{
    //         *p = image::Luma([(0)]);
    //     }else{
    //         *p = image::Luma([(125)]);
    //     }
    // });

    /* */

    let e = vec![12,3,3];
    // what is we use zip instead??
    let v = non.par_chunks(width).zip(non_max.par_chunks_mut(width)).into_par_iter().for_each(
        |(vec,img)|
            {
            vec.iter().zip((*img).iter_mut()).for_each(
                |(v1,m2)|{
                    let v1 = *v1;
                    if v1 > 0.7{
                        *m2 = 255
                    }else if v1 < 0.25{
                        *m2 = 0
                    }else{
                        *m2 = 50
                    }
                }
            )

        }
    );
    non_max.save("par_thres.png").unwrap();
}

fn hyst(thres: &mut GrayImage){
    let (width,height) = thres.dimensions();
    let mut n_img: GrayImage = ImageBuffer::new(width,height);
    let mut point = UnsafeSlice::new(n_img.as_parallel_slice_mut());
    for x in 1..width-1{
            (1..height-1).into_par_iter().for_each(|y|{
            let y1 =y;
            let y = y -1;
            let a1_1 = thres.get_pixel(x-1, y).0[0];
            let a2_1 = thres.get_pixel(x, y).0[0];
            let a3_1 = thres.get_pixel(x+1, y).0[0];
            let y = y +1;
            let a1_2 = thres.get_pixel(x-1, y).0[0];
            let a2_2 = thres.get_pixel(x, y).0[0];
            let a3_2 = thres.get_pixel(x+1, y).0[0];
            let y= y +1;
            let a1_3 = thres.get_pixel(x-1, y).0[0];
            let a2_3 = thres.get_pixel(x, y).0[0];
            let a3_3 = thres.get_pixel(x+1, y).0[0];
            let a = vec![a1_1,a2_1,a3_1,a1_2,a3_2,a1_3,a2_3,a3_3];
            let a:Vec<u8> = a.iter().map(|f| *f).filter(|p| *p >= 50).collect();
            if a.len() >= 2 && (a2_2 == 50 || a2_2 == 255) {
                //*n_img.get_pixel_mut(x, y1) = image::Luma([255]);
                unsafe{point.write((x + width*y1) as usize, 255);}
            }else{
                //*n_img.get_pixel_mut(x, y1) = image::Luma([0]);
                unsafe{point.write((x + width*y1) as usize, 0);}
            }

        });
    }
    n_img.save("par_hyst4.png").unwrap();

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

/*
(1..width-1).into_iter().for_each(|x|{
        (1..height-1).into_par_iter().for_each(|y|{
            let y1 =y;
            let y = y -1;
            let a1_1 = thres.get_pixel(x-1, y).0[0];
            let a2_1 = thres.get_pixel(x, y).0[0];
            let a3_1 = thres.get_pixel(x+1, y).0[0];
            let y = y +1;
            let a1_2 = thres.get_pixel(x-1, y).0[0];
            let a2_2 = thres.get_pixel(x, y).0[0];
            let a3_2 = thres.get_pixel(x+1, y).0[0];
            let y= y +1;
            let a1_3 = thres.get_pixel(x-1, y).0[0];
            let a2_3 = thres.get_pixel(x, y).0[0];
            let a3_3 = thres.get_pixel(x+1, y).0[0];
            let a = vec![a1_1,a2_1,a3_1,a1_2,a3_2,a1_3,a2_3,a3_3];
            let a:Vec<u8> = a.iter().map(|f| *f).filter(|p| *p >= 50).collect();
            if a.len() >= 1 && (a2_2 == 50 || a2_2 == 255) {
                *n_img.get_pixel_mut(x, y1) = image::Luma([255]);
            }else{
                *n_img.get_pixel_mut(x, y1) = image::Luma([0]);
            }

        });
    });
 */