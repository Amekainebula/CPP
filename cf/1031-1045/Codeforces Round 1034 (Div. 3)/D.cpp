#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 1e6 + 6;
void Murasame()
{
    // 11011
    // 01010
    // 11010
    // 01000
    // 01110

    //111111
    //010100
    //011111
    //000100
    //111100
    //
    //6 3
    //111111
    //010101
    //111101
    //010100
    //111100
    //010000
    //011110
    //000100
    //
    //7 4
    //1111111
    //0101010
    //1111010
    //0001000
    
    int n, k;
    cin >> n >> k;
    string s;
    cin >> s;
    if(k>n/2)
    {
        cout<<"Alice"<<endl;
        return;
    }
    int cnt = 0;
    ff(i, 0, n-1)
    {
        if(s[i]=='1')
        cnt++;
    }
    if(cnt<=k)
    {
        cout<<"Alice"<<endl;
        return;
    }
    cout<<"Bob"<<endl;
    // s = 'l' + s;
    // vi a(n + 1);
    // int cnt = n - k + 1;
    // ff(i, 1, k)
    // {
    //     if (s[i] == '1')
    //         a[1]++;
    // }
    // ff(i, 2, cnt)
    // {
    //     a[i] = a[i - 1] + (s[i + k - 1] - '0') - (s[i - 1] - '0');
    // }
    // ff(i, 1, cnt)
    // {
    //     cout << a[i] << ' ';
    // }
    // cout << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}