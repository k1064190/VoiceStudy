import os

def phone_to_int(label_str, label_int,
                 phone_list, insert_sil=False):
    with open(label_str, 'r') as f_in, open(label_int, 'w') as f_out:
        for line in f_in:
            text = line.split()
            f_out.write("%s " % text[0])
            if insert_sil:
                f_out.write("0 ")
            for phone in text[1:]:
                f_out.write(" %d " % phone_list.index(phone))
            if insert_sil:
                f_out.write(" 0")
            f_out.write("\n")


if __name__=="__main__":
    label_str = '../data/label/train_small/text_phone'
    out_dir = './exp/data/train_small'
    phone_file = './phones.txt'
    silence_phone = 'pau'
    insert_sil = True

    phone_list = [silence_phone]
    with open(phone_file, 'r') as f:
        for line in f:
            phone_list.append(line.strip())
    out_phone_list = os.path.join(out_dir, 'phone_list')
    with open(out_phone_list, 'w+') as f:
        for i, phone in enumerate(phone_list):
            f.write("%s %d\n" % (phone, i))
    label_int = os.path.join(out_dir, 'text_int')
    phone_to_int(label_str, label_int, phone_list, insert_sil)
