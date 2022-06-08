use std::sync::mpsc;
use std::time::Duration;

pub struct AsyncTimerEvent<T> where T: Send + Default + 'static {
    pub item: T,
    receiver: Option<mpsc::Receiver<T>>,
    sleep_time: u64,
}

impl<T> AsyncTimerEvent<T> where T: Send + Default + 'static {
    pub fn new(sleep_sec: u64) -> AsyncTimerEvent<T> {
        AsyncTimerEvent {
            item: T::default(),
            receiver: None,
            sleep_time: sleep_sec,
        }
    }

    #[inline(always)]
    pub fn check_buffer(&mut self) -> Option<T> {
        if let Some(recv) = self.receiver.as_ref() {
            if let Ok(result) = recv.try_recv() {
                self.deactivate();
                return Some(result);
            }
        }
        None
    } 

    pub fn activate(&mut self, reset_item: T, now_item: T) {
        self.item = now_item;
        if let Some(recv) = self.receiver.take() {
            drop(recv);             // if receiver exists, drop it
        }
        let (tx, rx) = mpsc::channel::<T>();
        self.receiver = Some(rx);
        let sleep_sec = self.sleep_time;
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(sleep_sec));
            let _ = tx.send(reset_item);          // whether successful or not, dropping tx (multiple calls to activate might incur problem (todo))
            drop(tx);
        });
    }

    fn deactivate(&mut self) {
        self.receiver = None;
    }
}