from uszipcode import ZipcodeSearchEngine
import re
search = ZipcodeSearchEngine()
res = search.by_coordinate(39.122229, -77.133578, radius=30)
print(res[0])


col_nm = '持股数(（万股）'
col_nm = re.split(r'\(|（', col_nm)[0].strip()
print(col_nm)


rs = re.match(r'\d+', '12个月')
print(rs[0])