library(Cairo)

data<-read.table("D:\\config\\time.txt",header=T)
x=t(data[1])
y1=t(data[2])
y2=t(data[3])
y3=t(data[4])
y4=t(data[5])

CairoPNG(file="D:\\config\\time.png",units="in",bg="white",width=5.5,height=5,dpi=300)

plot(x,y4,type = "o",xlab = "迭代次数",ylab = "运行时间",col="green",pch=c(15),family='STXihei')
lines(x,y2,type = "o",col="yellow",pch=c(16))
lines(x,y3,type = "o",col="blue",pch=c(17))
lines(x,y1,type = "o",col="red",pch=c(18))
legend("topleft",c("Base_CNN","Att_CNN1","Att_CNN2","Att_CNN3"),col=c("red","yellow","blue","green"),pch=c(18,16,17,15))
dev.off()
