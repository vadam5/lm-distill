import logging
import torch.optim as optim 

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from distiller import Distiller

class DistillTrainer():
    def __init__(
        self, 
        run_name,
        teacher,
        student,
        teacher_tokenizer,
        student_tokenizer,
        same_vocab=True,
        vocab_prob_map=None,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        lr=1e-4,
        epochs=20
    ):
        self.same_vocab = same_vocab
        self.distiller = Distiller(
            teacher=teacher, 
            student=student,
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
            same_vocab=same_vocab,
            vocab_prob_map=vocab_prob_map,
        )
        
        self.train_set = train_dataloader
        self.val_set = val_dataloader
        self.test_set = test_dataloader
        self.optimizer = optim.AdamW(self.distiller.parameters(), lr=lr)
        self.logger = logging.logger
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f'runs/{run_name}_{timestamp}')
    
    def step(self, inputs, train=True):
        if self.same_vocab:
            teacher_inputs = inputs
            student_inputs = inputs
        else:
            teacher_inputs, student_inputs = inputs
        
        self.optimizer.zero_grad()
        teacher_outputs, student_outputs = self.distiller(teacher_inputs, student_inputs)
        teacher_logits = teacher_outputs["logits"]
        teacher_preds = teacher_outputs["sentences"]
        student_logits = student_outputs["logits"]
        student_preds = student_outputs["sentences"]
        
        loss, student_loss, distill_loss = self.distiller.loss_fn(teacher_logits, teacher_preds, student_logits, student_preds)
        
        if train:
            loss.backward()
            self.optimizer.step()
            
        return loss, student_loss, distill_loss

    def epoch(self, epoch_num, log_every=1, val_every=0.20):
        val_at = int(len(self.train_set) * 0.20)
        
        for step_num, inputs in enumerate(self.train_set):
            train_loss, train_student_loss, train_distill_loss = self.train_step(inputs)
            
            if step_num % log_every == 0:
                self.loss_logger(epoch_num, step_num, train_loss, train_student_loss, train_distill_loss)
            
            if step_num % val_at ==  0:
                self.distiller.train(False)
                running_loss = 0
                running_student_loss = 0
                running_distill_loss = 0
                
                for inputs in self.val_set:
                    loss, student_loss, distill_loss = self.step(inputs, train=False)
                    running_loss += loss
                    running_student_loss += student_loss
                    running_distill_loss += distill_loss
                    
                val_loss = running_loss / len(self.val_set)
                val_student_loss = running_student_loss / len(self.val_set)
                val_distill_loss = running_distill_loss / len(self.val_set)
                
                self.loss_logger(epoch_num, step_num, val_loss, val_student_loss, val_distill_loss, train=False)
                self.distiller.train(True)
                
    def loss_logger(self, epoch_num, step_num, loss, student_loss, distill_loss, train=True):
        stage = "Train"
        if not train:
            stage = "Val"
            
        self.logger.info(f"Epoch: {epoch_num} | Train Step Num {step_num} | Total {stage} Loss: {loss} | Student {stage} Loss: {student_loss} | Distill {stage} Loss: {distill_loss}")
            
        global_step_num = (epoch_num * len(self.train_set)) + step_num + 1
        self.writer.add_scalar(f'Total_Loss/{stage}', loss, global_step_num)
        self.writer.add_scalar(f'Student_Loss/{stage}', student_loss, global_step_num)
        self.writer.add_scalar(f'Distill_Loss/{stage}', distill_loss, global_step_num)
                
    def train(self):
        pass
    
    def val(self):
        pass
    
    def test(self):
        pass
    
    def save(self):
        pass
    
    
    