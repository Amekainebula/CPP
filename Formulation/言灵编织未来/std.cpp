//https://www.luogu.com.cn/record/261957723
// 平均用时4.7ms，最快用时4ms，最慢用时6ms
// 内存使用1.23mb
#include <bits/stdc++.h>

using namespace std;
using i64 = long long;

string F(string t, int sum)
{
    if (sum % 2 == 1 || sum < 0)
    {
        // cerr << "Error" << endl; 
    }
    string res = "";
    sum >>= 1;
    while (sum)
    {
        int x = min(26, sum);
        sum -= x;
        res += char(x + 'a' - 1);
    }
    string rvres(res.rbegin(), res.rend());
    return rvres + t + res;
}
void solve()
{
    string s, t = "";
    cin >> s;
    int sum = 0;
    for (int i = 0; i < s.size(); i++)
    {
        sum += s[i] - 'a' + 1;
    }
    // int debug_sum = sum;
    string pre[2] = {"promisesimorp", "promiseesimorp"};
    int arr[2] = {185, 190};
    if (sum & 1)
    {
        t = pre[0];
        sum -= 185;
    }
    else
    {
        t = pre[1];
        sum -= 190;
    }
    // int debug_ok = 0;
    string ans = F(t, sum);
    // for (int i = 0; i < ans.size(); i++)
    // {
    //     debug_ok += ans[i] - 'a' + 1;
    // }

    cout << ans << '\n';
    // cerr << (debug_ok == debug_sum && ans.size() <= s.size() * 2 ? "YES" : "NO") << endl;
}
int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    int _T;
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}
