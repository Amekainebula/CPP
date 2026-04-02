#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define vi vector<int>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
void Murasame()
{
    int q;
    cin >> q;
    deque<int> f, b;
    int ans_f = 0, ans_b = 0;
    int n = 0;
    int sum = 0;
    while (q--)
    {
        int op, x;
        cin >> op;
        if (op == 1)
        {
            f.push_front(f.back());
            ans_f += sum;
            ans_f -= f.back() * n;
            f.pop_back();
            b.push_back(b.front());
            ans_b -= sum;
            ans_b += b.front() * n;
            b.pop_front();
        }
        else if (op == 2)
        {
            swap(f, b);
            swap(ans_f, ans_b);
        }
        else
        {
            n++;
            cin >> x;
            f.push_back(x);
            ans_f += x * n;
            b.push_front(x);
            sum += x;
            ans_b += sum;
        }
        cout << ans_f << endl;
    }
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