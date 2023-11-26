fn ss(s:&Vec<i32>) -> i32{
    s[0]

}


fn main() {
    println!("Hello, world!");
    let five: Vec<i32> = vec![123];
    let ans = ss(&five);
    println!("{ans}");

}
