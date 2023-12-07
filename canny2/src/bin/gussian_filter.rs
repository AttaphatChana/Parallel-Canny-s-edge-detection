use image::{DynamicImage, GenericImageView};



fn gus_filt(img:&DynamicImage) -> image::DynamicImage{
    //
    let (x,y) = img.dimensions();
    let img = img.as_rgb8()
    .into_iter().map(|p| p.p)

}

fn main(){

}