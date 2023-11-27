use image::{GenericImageView, DynamicImage};
use ndarray_image::{self, open_image};
pub struct Picture{
    pixel_ker: Vec<Vec<u8>>


    
}

fn gus_filt(img: &DynamicImage){
    
}

fn get_str_ascii(intent :u8)-> &'static str{
    let index = intent/32;
    let ascii = [" ",".",",","-","~","+","=","@"];
    return ascii[index as usize];
}

fn get_image(dir: &str, scale: u32){
    let img = image::open(dir).unwrap();
    println!("{:?}", img.dimensions());
    let (width,height) = img.dimensions();
    for y in 0..height{
        for x in 0..width{
            if y % (scale * 2) == 0 && x % scale ==0{
                let pix = img.get_pixel(x,y);
                let mut intent = pix[0]/3 + pix[1]/3 + pix[2]/3;
                if pix[3] ==0{
                    intent = 0;
                }
                print!("{}",get_str_ascii(intent));
            } 
        }
        if y%(scale*2)==0{
            println!("");
        }
    }
}

fn main() {
    println!("Hello, world!");
    let img:image::ImageBuffer<image::Luma<u8>, Vec<u8>> = image::open("../brick-house.png").unwrap().into_luma8();
    let img2 = open_image("../brick-house.png", ndarray_image::Colors::Bgra).expect("unable to open input image");
    println!("{:?}",img[(0,10)]);
}
